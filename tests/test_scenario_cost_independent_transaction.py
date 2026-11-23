import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

from proteus import Model
from trytond.model.exceptions import ValidationError
from trytond.pool import Pool
from trytond.tests.test_tryton import DB_NAME, drop_db
from trytond.tests.tools import activate_modules
from trytond.transaction import Transaction


class TestCostIndependentTransaction(unittest.TestCase):

    def setUp(self):
        drop_db()
        super().setUp()

    def tearDown(self):
        drop_db()
        super().tearDown()

    def test(self):
        activate_modules('ai_model')
        ClientModel = Model.get('ai.model')
        saved = ClientModel(name='Committed fallback',
            model_name='cost/fallback', provider='openrouter', type='llm')
        saved.save()

        with Transaction().start(DB_NAME, 1) as transaction:
            AIModel = Pool().get('ai.model')
            model = AIModel.get_or_create('cost/model')
            model_id = model.id
            transaction.rollback()
            model = AIModel(model_id)
            self.assertEqual(model.model_name, 'cost/model')
            self.assertEqual(
                AIModel.get_or_create('cost/model').id, model_id)

            embedding = AIModel.get_or_create(
                'openai/text-embedding-3-small', type_='embedding')
            embedding_id = embedding.id
            transaction.rollback()
            embedding = AIModel(embedding_id)
            self.assertEqual(embedding.type, 'embedding')

            response = SimpleNamespace(
                usage=SimpleNamespace(
                    cost='0.00042', prompt_tokens=120,
                    completion_tokens=30),
                choices=[SimpleNamespace(
                        message=SimpleNamespace(content='Completion'))])
            client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(
                        create=lambda **kwargs: response)))

            result, error = model.get_completion(
                [{'role': 'user', 'content': 'Hello'}], model,
                client=client)
            AIModel.write([model], {'name': 'Rolled back name'})
            transaction.rollback()

            self.assertFalse(error)
            self.assertIs(result, response)

            embedding_response = SimpleNamespace(
                usage=SimpleNamespace(cost='0.00003', prompt_tokens=12),
                data=[SimpleNamespace(embedding=[0.1, 0.2])])
            embedding_client = SimpleNamespace(
                embeddings=SimpleNamespace(
                    create=lambda **kwargs: embedding_response))
            self.assertIs(embedding.get_embeddings(
                    'Text to embed', model, client=embedding_client),
                embedding_response)
            transaction.rollback()

        with Transaction().start(DB_NAME, 1):
            AIModel = Pool().get('ai.model')
            Cost = Pool().get('ai.model.cost')
            model = AIModel(model_id)
            cost, = Cost.search([('model', '=', model_id)])

            self.assertEqual(model.name, 'cost/model')
            self.assertEqual(cost.origin, str(model))
            self.assertEqual(cost.model, model)
            self.assertEqual(cost.input_tokens, 120)
            self.assertEqual(cost.output_tokens, 30)
            embedding_cost, = Cost.search([('model', '=', embedding_id)])
            self.assertEqual(embedding_cost.origin, str(model))
            self.assertEqual(embedding_cost.input_tokens, 12)

        with Transaction().start(DB_NAME, 1) as transaction:
            Action = Pool().get('ir.action.act_window')
            Cost = Pool().get('ai.model.cost')
            model = Pool().get('ai.model')(model_id)
            origin, = Action.create([{
                        'name': 'New cost origin',
                        'res_model': 'ai.model',
                        }])
            origin_reference = str(origin)
            # SQLite shares a connection between transactions. Simulate the
            # visibility of an uncommitted origin from a separate connection.
            with patch.object(Action, 'search', return_value=[]):
                result, error = model.get_completion(
                    [{'role': 'user', 'content': 'Hello'}], origin,
                    client=client, parent_origin=model)
                cost, = Cost.search([
                    ('origin', '=', origin_reference)])
                self.assertIsNone(cost.origin)
                self.assertEqual(cost.parent_origin, model)
                self.assertIn(cost,
                    Cost.search([('parent_origin', '=', str(model))]))
            self.assertFalse(error)
            self.assertIs(result, response)
            transaction.commit()

        with Transaction().start(DB_NAME, 1) as transaction:
            Cost = Pool().get('ai.model.cost')
            cost, = Cost.search([('origin', '=', origin_reference)])
            cost_id = cost.id
            self.assertEqual(cost.origin, origin_reference)
            self.assertEqual(cost.input_tokens, 120)
            self.assertFalse(Cost.search([('id', '=', cost_id),
                ('origin', '=', str(model))]))
            # The independently saved cost must also remain readable when
            # its origin disappears, as after a caller rollback or deletion.
            Action.delete([Action(origin.id)])
            transaction.commit()

        with Transaction().start(DB_NAME, 1):
            cost = Pool().get('ai.model.cost')(cost_id)
            model = Pool().get('ai.model')(model_id)
            self.assertIsNone(cost.origin)
            self.assertEqual(cost.parent_origin, model)
            self.assertEqual(cost.input_tokens, 120)
            self.assertEqual(cost.origin_reference, origin_reference)

            # Search the original reference after deletion.
            for domain, expected in [
                    ([('origin', '=', origin_reference)], True),
                    ([('origin', '!=', origin_reference)], False),
                    ([('origin', '=', str(model))], False),
                    ([('origin', '!=', str(model))], True),
                    ([('origin', 'in', [origin_reference])], True),
                    ([('origin', 'not in', [origin_reference])], False),
                    ([('origin', '=', None)], False),
                    ([('origin', '!=', None)], True)]:
                with self.subTest(domain=domain):
                    self.assertEqual(bool(Cost.search(
                        [('id', '=', cost_id)] + domain)), expected)

            # Invalid model names and unallocated IDs fail before any API use.
            with self.assertRaises(ValidationError):
                model.get_completion([], Pool().get('ai.model')(),
                    parent_origin=model, client=client)
            with patch.object(Cost, 'get_models',
                    return_value=[('ai.model', 'Model')]):
                with self.assertRaises(ValidationError):
                    Cost.check_origin_reference(origin_reference)
            for value in ('ai.model,None', 'ai.model,-1', 'ai.model,abc',
                    'unknown.model,1'):
                with self.subTest(reference=value):
                    with self.assertRaises(ValidationError):
                        Cost.check_origin_reference(value)

            Action = Pool().get('ir.action.act_window')
            new_origin, = Action.create([{
                'name': 'Uncommitted fallback', 'res_model': 'ai.model'}])
            with patch.object(Action, 'search', return_value=[]):
                with self.assertRaisesRegex(ValueError, 'must be committed'):
                    model.get_completion([], model,
                        parent_origin=new_origin, client=client)

        # Cost fields are readonly to clients; validate the internal create
        # path, then inspect its outcome through Proteus.
        values = {'model': saved.id,
            'parent_origin': str(saved),
            'duration': timedelta(seconds=1)}
        with Transaction().start(DB_NAME, 1) as transaction:
            Cost = Pool().get('ai.model.cost')
            with self.assertRaisesRegex(ValidationError, 'Invalid AI cost origin'):
                Cost.create([values | {'origin_reference': 'unknown.model,1'}])
            transaction.rollback()
        with Transaction().start(DB_NAME, 1) as transaction:
            Cost = Pool().get('ai.model.cost')
            missing, parent_only, inactive = Cost.create([
                values | {'origin_reference': 'ai.model,2147483647'},
                values,
                values | {'origin_reference': str(saved)}])
            missing_id, parent_only_id, inactive_id = (
                missing.id, parent_only.id, inactive.id)
            transaction.commit()
        ClientCost = Model.get('ai.model.cost')
        missing = ClientCost(missing_id)
        self.assertIsNone(missing.origin)
        parent_only = ClientCost(parent_only_id)
        self.assertIsNone(parent_only.origin)
        self.assertEqual(parent_only.parent_origin, saved)
        self.assertIn(parent_only, ClientCost.find([('origin', '=', None)]))
        saved.active = False
        saved.save()
        missing.reload()
        self.assertEqual(ClientCost(inactive_id).origin, str(saved))
        self.assertEqual(missing.parent_origin, saved)
        self.assertIn(missing, ClientCost.find([
            ('origin', '=', 'ai.model,2147483647')]))
