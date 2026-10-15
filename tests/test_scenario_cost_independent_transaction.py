import unittest
from types import SimpleNamespace
from unittest.mock import patch

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
            self.assertEqual(cost.origin, model)
            self.assertEqual(cost.model, model)
            self.assertEqual(cost.input_tokens, 120)
            self.assertEqual(cost.output_tokens, 30)
            embedding_cost, = Cost.search([('model', '=', embedding_id)])
            self.assertEqual(embedding_cost.origin, model)
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
                    client=client)
            self.assertFalse(error)
            self.assertIs(result, response)
            transaction.commit()

        with Transaction().start(DB_NAME, 1) as transaction:
            Cost = Pool().get('ai.model.cost')
            cost, = Cost.search([('origin', '=', origin_reference)])
            cost_id = cost.id
            self.assertEqual(str(cost.origin), origin_reference)
            self.assertEqual(cost.input_tokens, 120)
            # The independently saved cost must also remain readable when
            # its origin disappears, as after a caller rollback or deletion.
            Action.delete([cost.origin])
            transaction.commit()

        with Transaction().start(DB_NAME, 1):
            cost = Pool().get('ai.model.cost')(cost_id)
            self.assertIsNone(cost.origin)
            self.assertEqual(cost.input_tokens, 120)
            table = cost.__table__()
            cursor = Transaction().connection.cursor()
            cursor.execute(*table.select(table.origin,
                    where=table.id == cost_id))
            self.assertEqual(cursor.fetchone()[0], origin_reference)
