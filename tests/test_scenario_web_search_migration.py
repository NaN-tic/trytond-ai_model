import unittest

from sql import Literal
from trytond.pool import Pool
from trytond.tests.test_tryton import DB_NAME, drop_db
from trytond.tests.tools import activate_modules
from trytond.transaction import Transaction


class TestWebSearchMigration(unittest.TestCase):

    def setUp(self):
        drop_db()
        super().setUp()

    def tearDown(self):
        drop_db()
        super().tearDown()

    def test(self):
        activate_modules('ai_model')

        with Transaction().start(DB_NAME, 1):
            AIModel = Pool().get('ai.model')
            native, prohibited = AIModel.create([{
                        'name': 'Native',
                        'model_name': 'example/native',
                        'provider': 'openrouter',
                        'type': 'llm',
                        }, {
                        'name': 'Prohibited',
                        'model_name': 'example/prohibited',
                        'provider': 'openrouter',
                        'type': 'llm',
                        }])
            table = AIModel.__table__()
            handler = AIModel.__table_handler__('ai_model')
            cursor = Transaction().connection.cursor()
            handler.column_rename(
                'allow_web_search', 'allow_web_search_selection')
            handler.add_column('allow_web_search', 'BOOL')
            cursor.execute(*table.update(
                    [table.allow_web_search], [Literal(True)],
                    where=table.id == native.id))
            cursor.execute(*table.update(
                    [table.allow_web_search], [Literal(False)],
                    where=table.id == prohibited.id))
            for cache in Transaction().cache.values():
                cache.clear()

            AIModel.__register__('ai_model')

            self.assertEqual(AIModel(native.id).allow_web_search, 'native')
            self.assertEqual(
                AIModel(prohibited.id).allow_web_search, 'prohibited')
            self.assertFalse(handler.column_exist('allow_web_search_boolean'))
