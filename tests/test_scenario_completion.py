import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

from sql import Table
from trytond import backend
from trytond.i18n import gettext
from trytond.model import fields
from trytond.modules.ai_model import completion
from trytond.modules.ai_model.ai import migrate_model_data
from trytond.pool import Pool
from trytond.tests.test_tryton import DB_NAME, drop_db
from trytond.tests.tools import activate_modules
from trytond.transaction import Transaction


class TestCompletion(unittest.TestCase):

    def setUp(self):
        drop_db()
        super().setUp()

    def tearDown(self):
        drop_db()
        super().tearDown()

    def test(self):
        activate_modules('ai_model')

        with Transaction().start(DB_NAME, 1):
            ModelData = Pool().get('ir.model.data')
            model_data, = ModelData.search([
                    ('module', '=', 'ai_model'),
                    ('fs_id', '=', 'view_ai_model_form'),
                    ])
            ModelData.write([model_data], {'module': 'nantic_connection'})
            migrate_model_data()
            migrated, = ModelData.search([
                    ('module', '=', 'ai_model'),
                    ('fs_id', '=', 'view_ai_model_form'),
                    ])
            self.assertEqual(migrated.id, model_data.id)

            intermediate_data, = ModelData.search([
                    ('module', '=', 'ai_model'),
                    ('fs_id', '=', 'menu_ai'),
                    ])
            ModelData.write(
                [intermediate_data], {'module': 'nantic_ai_model'})
            migrate_model_data()
            migrated, = ModelData.search([
                    ('module', '=', 'ai_model'),
                    ('fs_id', '=', 'menu_ai'),
                    ])
            self.assertEqual(migrated.id, intermediate_data.id)

            pool = Pool()
            OpenRouterModel = pool.get('ai.model.openrouter')
            AIModel = pool.get('ai.model')
            Configuration = pool.get('ai.configuration')
            ActionWindow = pool.get('ir.action.act_window')
            ModelAccess = pool.get('ir.model.access')
            Cost = pool.get('ai.model.cost')
            openrouter_model, = OpenRouterModel.create([{
                        'openrouter_id': 'example/model',
                        'name': 'Example model',
                        'prompt_price': '1',
                        'completion_price': '2',
                        'input_cache_read_price': '0.5',
                        }])
            ai_model, = AIModel.create([{
                        'name': 'Example model',
                        'model_name': 'example/model',
                        'openrouter_model': openrouter_model.id,
                        'provider': 'openrouter',
                        'type': 'llm',
                        'llm_pdf_engine': 'native',
                        }])
            Transaction()._locked_tables.add(Configuration._table)
            Configuration.create([{'default_llm': ai_model.id}])
            action, = ActionWindow.create([{
                        'name': 'Example Model',
                        'res_model': OpenRouterModel.__name__,
                        }])
            access, = ModelAccess.create([{
                        'model': OpenRouterModel.__name__,
                        'perm_read': True,
                        }])

            self.assertIn('default_llm', Configuration._fields)
            self.assertIsInstance(
                AIModel._fields['openrouter_model'], fields.Function)
            self.assertFalse(AIModel.__table_handler__(
                    'ai_model').column_exist('openrouter_model'))
            self.assertNotIn('assistant_model', Configuration._fields)
            self.assertTrue(hasattr(
                    Configuration, 'get_default_llm_client'))
            self.assertFalse(hasattr(
                    Configuration, 'get_assistant_client'))

            table_renames = [
                ('ai_model_openrouter', 'nantic_ai_model_openrouter'),
                ('ai_model', 'nantic_ai_model'),
                ('ai_configuration', 'nantic_ai_configuration'),
                ]
            for new_table, old_table in table_renames:
                backend.TableHandler.table_rename(new_table, old_table)

            cursor = Transaction().connection.cursor()
            ir_model = Table('ir_model')
            ir_model_field = Table('ir_model_field')
            ir_model_access = Table('ir_model_access')
            translations = Table('ir_translation')
            model_names = [
                ('ai.model.openrouter', 'nantic.ai.model.openrouter'),
                ('ai.model', 'nantic.ai.model'),
                ('ai.configuration', 'nantic.ai.configuration'),
                ]
            for new_name, old_name in model_names:
                cursor.execute(*ir_model.update(
                        [ir_model.name, ir_model.module],
                        [old_name, 'nantic_ai_model'],
                        where=ir_model.name == new_name))
                cursor.execute(*ir_model_field.update(
                        [ir_model_field.model, ir_model_field.module],
                        [old_name, 'nantic_ai_model'],
                        where=ir_model_field.model == new_name))
                cursor.execute(*ir_model_field.update(
                        [ir_model_field.relation], [old_name],
                        where=ir_model_field.relation == new_name))
                cursor.execute(*ir_model_access.update(
                        [ir_model_access.model], [old_name],
                        where=ir_model_access.model == new_name))
                action_table = ActionWindow.__table__()
                cursor.execute(*action_table.update(
                        [action_table.res_model], [old_name],
                        where=action_table.res_model == new_name))
                cursor.execute(*translations.select(
                        translations.id, translations.name,
                        where=translations.name.like(new_name + '%')))
                for translation_id, name in cursor.fetchall():
                    cursor.execute(*translations.update(
                            [translations.name, translations.module], [
                                old_name + name[len(new_name):],
                                'nantic_ai_model'],
                            where=translations.id == translation_id))

            OpenRouterModel.__register__('ai_model')
            AIModel.__register__('ai_model')
            Configuration.__register__('ai_model')

            self.assertEqual(AIModel(ai_model.id).name, 'Example model')
            self.assertEqual(
                Configuration(1).default_llm.id, ai_model.id)
            self.assertFalse(ActionWindow.search([
                        ('id', '=', action.id),
                        ]))
            self.assertFalse(ModelAccess.search([
                        ('id', '=', access.id),
                        ]))
            self.assertFalse(pool.get('ir.model').search([
                        ('name', 'like', 'nantic.ai.%'),
                        ]))
            self.assertFalse(pool.get('ir.model.field').search([
                        'OR',
                        ('model', 'like', 'nantic.ai.%'),
                        ('relation', 'like', 'nantic.ai.%'),
                        ]))

            usage = SimpleNamespace(
                cost='0.00042', cost_currency='USD', prompt_tokens=120,
                completion_tokens=30,
                prompt_tokens_details=SimpleNamespace(cached_tokens=20))
            response = SimpleNamespace(
                usage=usage,
                choices=[SimpleNamespace(
                        message=SimpleNamespace(content='Completion'))])
            request = {}
            client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(
                        create=lambda **kwargs: (
                            request.update(kwargs) or response))))

            result, error = ai_model.get_completion(
                [{'role': 'user', 'content': [{
                                'type': 'file',
                                'file': {
                                    'filename': 'document.pdf',
                                    'file_data': (
                                        'data:application/pdf;base64,UERG'),
                                    },
                                }]}], ai_model,
                client=client)

            self.assertFalse(error)
            self.assertIs(result, response)
            self.assertIs(result.usage, usage)
            cost, = Cost.search([])
            self.assertEqual(cost.origin, ai_model)
            self.assertEqual(cost.model, ai_model)
            self.assertEqual(cost.input_tokens, 120)
            self.assertEqual(cost.cached_input_tokens, 20)
            self.assertEqual(cost.output_tokens, 30)
            self.assertEqual(cost.cost, Decimal('0.00042000'))
            self.assertEqual(cost.currency, 'USD')
            self.assertGreaterEqual(cost.duration.total_seconds(), 0)
            self.assertEqual(request['extra_body'], {
                    'plugins': [{
                            'id': 'file-parser',
                            'pdf': {'engine': 'native'},
                            }],
                    })
            amount, currency = completion.get_completion_cost(
                SimpleNamespace(usage=SimpleNamespace(
                        prompt_tokens=100, completion_tokens=50)),
                ai_model)
            self.assertIsNone(amount)
            self.assertEqual(currency, 'USD')

            default_cost_access, = ModelAccess.search([
                    ('model', '=', Cost.__name__),
                    ('group', '=', None),
                    ])
            self.assertFalse(default_cost_access.perm_read)
            self.assertFalse(default_cost_access.perm_write)
            self.assertFalse(default_cost_access.perm_create)
            self.assertFalse(default_cost_access.perm_delete)
            admin_group = ModelData.get_id('res', 'group_admin')
            cost_access, = ModelAccess.search([
                    ('model', '=', Cost.__name__),
                    ('group', '=', admin_group),
                    ])
            self.assertTrue(cost_access.perm_read)
            self.assertFalse(cost_access.perm_write)
            self.assertFalse(cost_access.perm_create)
            self.assertFalse(cost_access.perm_delete)

            with self.assertRaises(TypeError):
                ai_model.get_completion(
                    [{'role': 'user', 'content': 'Hello'}], client=client)

            automatic_model = AIModel.get_or_create('example/automatic')
            self.assertEqual(automatic_model.provider, 'openrouter')
            self.assertEqual(automatic_model.model_name, 'example/automatic')
            self.assertEqual(
                AIModel.get_or_create('example/automatic').id,
                automatic_model.id)
            automatic_response = SimpleNamespace(
                usage=SimpleNamespace(
                    cost='0.00123', prompt_tokens=40,
                    completion_tokens=5),
                choices=[SimpleNamespace(
                        message=SimpleNamespace(content='Automatic'))])
            automatic_client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(
                        create=lambda **kwargs: automatic_response)))
            automatic_model.get_completion(
                [{'role': 'user', 'content': 'Hello'}], ai_model,
                client=automatic_client)
            automatic_cost, = Cost.search([
                    ('model', '=', automatic_model.id),
                    ])
            self.assertIsNone(automatic_model.openrouter_model)
            self.assertEqual(automatic_cost.cost, Decimal('0.00123000'))
            automatic_openrouter_model, = OpenRouterModel.create([{
                        'openrouter_id': 'example/automatic',
                        'name': 'Automatic model',
                        }])
            self.assertEqual(
                AIModel(automatic_model.id).openrouter_model,
                automatic_openrouter_model)

            setter_model, = AIModel.create([{
                        'name': 'Setter model',
                        'model_name': 'manual',
                        'provider': 'openrouter',
                        'type': 'llm',
                        }])
            AIModel.write([setter_model], {
                    'openrouter_model': openrouter_model.id,
                    })
            setter_model = AIModel(setter_model.id)
            self.assertEqual(
                setter_model.model_name, openrouter_model.openrouter_id)
            self.assertEqual(setter_model.openrouter_model, openrouter_model)

            embedding_model = AIModel.get_or_create(
                'openai/text-embedding-3-small', type_='embedding')
            embedding_response = SimpleNamespace(
                usage=SimpleNamespace(
                    cost='0.00003', prompt_tokens=12, total_tokens=12),
                data=[SimpleNamespace(embedding=[0.1, 0.2])])
            embedding_client = SimpleNamespace(
                embeddings=SimpleNamespace(
                    create=lambda **kwargs: embedding_response))
            result = embedding_model.get_embeddings(
                'Text to embed', ai_model, client=embedding_client)
            self.assertIs(result, embedding_response)
            embedding_cost, = Cost.search([
                    ('model', '=', embedding_model.id),
                    ])
            self.assertEqual(embedding_cost.origin, ai_model)
            self.assertEqual(embedding_cost.input_tokens, 12)
            self.assertEqual(embedding_cost.cost, Decimal('0.00003000'))
            self.assertGreaterEqual(
                embedding_cost.duration.total_seconds(), 0)

            failing_client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(
                        create=lambda **kwargs: self.fail_completion())))
            with patch.object(completion.time, 'sleep'):
                result, error = ai_model.get_completion(
                    [{'role': 'user', 'content': 'Hello'}], ai_model,
                    client=failing_client)

            self.assertTrue(error)
            self.assertEqual(result, gettext(
                    'ai_model.msg_internal_server_error'))

    @staticmethod
    def fail_completion():
        raise RuntimeError('Completion failed')
