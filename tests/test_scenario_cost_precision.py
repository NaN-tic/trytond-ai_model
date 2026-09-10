import unittest
from datetime import timedelta
from decimal import Decimal

from trytond.pool import Pool
from trytond.tests.test_tryton import DB_NAME, drop_db
from trytond.tests.tools import activate_modules
from trytond.transaction import Transaction


class TestCostPrecision(unittest.TestCase):

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
            Cost = pool.get('ai.model.cost')
            model, = AIModel.create([{
                        'name': 'Cost model',
                        'model_name': 'cost/model',
                        'provider': 'openrouter',
                        'type': 'llm',
                        }])
            cost, = Cost.create([{
                        'origin': str(model),
                        'model': model.id,
                        'cost': Decimal('0.124971318'),
                        'duration': timedelta(),
                        }])

            self.assertEqual(cost.cost, Decimal('0.124971318'))
