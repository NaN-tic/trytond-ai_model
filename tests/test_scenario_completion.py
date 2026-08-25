import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sql import Table
from trytond import backend
from trytond.i18n import gettext
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
            openrouter_model, = OpenRouterModel.create([{
                        'openrouter_id': 'example/model',
                        'name': 'Example model',
                        }])
            ai_model, = AIModel.create([{
                        'name': 'Example model',
                        'model_name': 'example/model',
                        'openrouter_model': openrouter_model.id,
                        'provider': 'openrouter',
                        'type': 'llm',
                        }])
            Transaction()._locked_tables.add(Configuration._table)
            Configuration.create([{'assistant_model': ai_model.id}])
            action, = ActionWindow.create([{
                        'name': 'Example Model',
                        'res_model': OpenRouterModel.__name__,
                        }])
            access, = ModelAccess.create([{
                        'model': OpenRouterModel.__name__,
                        'perm_read': True,
                        }])

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
                Configuration(1).assistant_model.id, ai_model.id)
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

            usage = SimpleNamespace(cost='0.00042', cost_currency='USD')
            response = SimpleNamespace(
                usage=usage,
                choices=[SimpleNamespace(
                        message=SimpleNamespace(content='Completion'))])
            client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(
                        create=lambda **kwargs: response)))

            result, error = completion.get_completion(
                [{'role': 'user', 'content': 'Hello'}],
                model='test', client=client)

            self.assertFalse(error)
            self.assertIs(result, response)
            self.assertIs(result.usage, usage)

            failing_client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(
                        create=lambda **kwargs: self.fail_completion())))
            with patch.object(completion.time, 'sleep'):
                result, error = completion.get_completion(
                    [{'role': 'user', 'content': 'Hello'}],
                    model='test', client=failing_client)

            self.assertTrue(error)
            self.assertEqual(result, gettext(
                    'ai_model.msg_internal_server_error'))

    @staticmethod
    def fail_completion():
        raise RuntimeError('Completion failed')
