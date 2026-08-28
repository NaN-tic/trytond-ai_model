from trytond.pool import Pool

from . import ai

__all__ = ['register']


def register():
    Pool.register(
        ai.Cron,
        ai.OpenRouterModel,
        ai.AIModel,
        ai.AIModelCost,
        ai.AIConfiguration,
        module='ai_model', type_='model')
