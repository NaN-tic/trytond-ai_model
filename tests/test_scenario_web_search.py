import unittest
from types import SimpleNamespace

from trytond.model.modelstorage import SelectionValidationError
from trytond.pool import Pool
from trytond.tests.test_tryton import DB_NAME, drop_db
from trytond.tests.tools import activate_modules
from trytond.transaction import Transaction


class TestWebSearch(unittest.TestCase):

    def setUp(self):
        drop_db()
        super().setUp()

    def tearDown(self):
        drop_db()
        super().tearDown()

    def test(self):
        activate_modules('ai_model')

        with Transaction().start(DB_NAME, 1):
            pool = Pool()
            AIModel = pool.get('ai.model')
            OpenRouterModel = pool.get('ai.model.openrouter')

            unsupported, supported = OpenRouterModel.create([{
                        'openrouter_id': 'example/unsupported',
                        'name': 'Unsupported',
                        'supported_parameters': 'temperature',
                        }, {
                        'openrouter_id': 'example/supported',
                        'name': 'Supported',
                        'supported_parameters': (
                            'temperature, web_search_options'),
                        }])

            unsupported_model = AIModel(
                name='Unsupported', type='llm', provider='openrouter',
                model_name=unsupported.openrouter_id)
            self.assertEqual(
                unsupported_model.get_web_search_options(),
                [('prohibited', 'Prohibited')])
            unsupported_model.allow_web_search = 'native'
            with self.assertRaises(SelectionValidationError):
                unsupported_model.save()

            supported_model = AIModel(
                name='Supported', type='llm', provider='openrouter',
                model_name=supported.openrouter_id,
                allow_web_search='native')
            self.assertIn(
                ('native', 'Native'),
                supported_model.get_web_search_options())
            supported_model.save()

            request = {}
            response = SimpleNamespace(
                usage=SimpleNamespace(
                    cost='0.0001', prompt_tokens=1, completion_tokens=1),
                choices=[SimpleNamespace(
                        message=SimpleNamespace(content='Result'))])
            client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(
                        create=lambda **kwargs: (
                            request.update(kwargs) or response))))
            result, error = supported_model.get_completion(
                [{'role': 'user', 'content': 'Search'}], supported_model,
                client=client)

            self.assertFalse(error)
            self.assertIs(result, response)
            self.assertEqual(request['web_search_options'], {})

