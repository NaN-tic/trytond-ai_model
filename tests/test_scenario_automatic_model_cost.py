import unittest
from decimal import Decimal
from types import SimpleNamespace

from proteus import Model
from trytond.pool import Pool
from trytond.tests.test_tryton import DB_NAME, drop_db
from trytond.tests.tools import activate_modules
from trytond.transaction import Transaction, TransactionError


class TestAutomaticModelCost(unittest.TestCase):

    def setUp(self):
        drop_db()
        super().setUp()

    def tearDown(self):
        drop_db()
        super().tearDown()

    def test(self):
        activate_modules('ai_model')
        AIModel = Model.get('ai.model')
        origin = AIModel(
            name='Cost origin', model_name='example/origin',
            provider='openrouter', type='llm')
        origin.save()

        response = SimpleNamespace(
            usage=SimpleNamespace(
                cost='0.00042', prompt_tokens=120, completion_tokens=30),
            choices=[SimpleNamespace(
                    message=SimpleNamespace(content='Completion'))],
            data=[SimpleNamespace(embedding=[0.1, 0.2])])
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(
                    create=lambda **kwargs: response)),
            embeddings=SimpleNamespace(create=lambda **kwargs: response))

        for type_ in ('llm', 'embedding'):
            with self.subTest(type=type_):
                model_name = f'example/automatic-{type_}'
                with Transaction().start(DB_NAME, 1) as transaction:
                    ServerModel = Pool().get('ai.model')
                    try:
                        model = ServerModel.get_or_create(
                            model_name, type_=type_)
                    except TransactionError:
                        # Emulate the RPC retry with a fresh database snapshot.
                        transaction.rollback()
                        model = ServerModel.get_or_create(
                            model_name, type_=type_)
                    model_id = model.id
                    self.assertEqual(ServerModel.get_or_create(
                            model_name, type_=type_).id, model_id)

                    with transaction.new_transaction(readonly=True):
                        visible, = ServerModel.search([
                                ('model_name', '=', model_name),
                                ('type', '=', type_),
                                ])
                        self.assertEqual(visible.id, model_id)

                    saved_origin = ServerModel(origin.id)
                    if type_ == 'llm':
                        result, error = model.get_completion(
                            [{'role': 'user', 'content': 'Hello'}],
                            saved_origin, client=client)
                        self.assertFalse(error)
                    else:
                        result = model.get_embeddings(
                            'Hello', saved_origin, client=client)
                    self.assertIs(result, response)
                    ServerModel.write(
                        [saved_origin], {'name': 'Rolled back name'})
                    transaction.rollback()

                with Transaction().start(DB_NAME, 1):
                    ServerModel = Pool().get('ai.model')
                    Cost = Pool().get('ai.model.cost')
                    model, = ServerModel.search([
                            ('model_name', '=', model_name),
                            ('type', '=', type_),
                            ])
                    cost, = Cost.search([('model', '=', model.id)])
                    self.assertEqual(model.id, model_id)
                    self.assertEqual(cost.origin.id, origin.id)
                    self.assertEqual(cost.origin.name, 'Cost origin')
                    self.assertEqual(cost.input_tokens, 120)
                    self.assertEqual(cost.output_tokens, 30)
                    self.assertEqual(cost.cost, Decimal('0.00042000'))
