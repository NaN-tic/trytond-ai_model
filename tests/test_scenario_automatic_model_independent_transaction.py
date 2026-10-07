import unittest
from types import SimpleNamespace

from trytond.pool import Pool
from trytond.tests.test_tryton import DB_NAME, drop_db
from trytond.tests.tools import activate_modules
from trytond.transaction import Transaction


class TestAutomaticModelCost(unittest.TestCase):

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
            origin, = AIModel.create([{
                        'name': 'Origin model',
                        'model_name': 'origin/model',
                        'provider': 'openrouter',
                        'type': 'llm',
                        }])
            transaction.commit()

            automatic_model = AIModel.get_or_create('automatic/model')
            response = SimpleNamespace(
                usage=SimpleNamespace(
                    cost='0.00042', prompt_tokens=120,
                    completion_tokens=30),
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content='Completion'))])
            client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(
                    create=lambda **kwargs: response)))

            result, error = automatic_model.get_completion(
                [{'role': 'user', 'content': 'Hello'}], origin,
                client=client)
            transaction.commit()

            self.assertFalse(error)
            self.assertIs(result, response)

        with Transaction().start(DB_NAME, 1):
            AIModel = Pool().get('ai.model')
            Cost = Pool().get('ai.model.cost')
            automatic_model, = AIModel.search([
                    ('model_name', '=', 'automatic/model'),
                    ])
            cost, = Cost.search([
                    ('model', '=', automatic_model.id),
                    ])

            self.assertEqual(cost.model, automatic_model)
