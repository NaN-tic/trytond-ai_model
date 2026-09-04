import unittest
from types import SimpleNamespace

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
            model, = AIModel.create([{
                        'name': 'Cost model',
                        'model_name': 'cost/model',
                        'provider': 'openrouter',
                        'type': 'llm',
                        }])
            transaction.commit()

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

        with Transaction().start(DB_NAME, 1):
            AIModel = Pool().get('ai.model')
            Cost = Pool().get('ai.model.cost')
            model, = AIModel.search([])
            cost, = Cost.search([])

            self.assertEqual(model.name, 'Cost model')
            self.assertEqual(cost.origin, model)
            self.assertEqual(cost.model, model)
            self.assertEqual(cost.input_tokens, 120)
            self.assertEqual(cost.output_tokens, 30)
