pool = globals()['pool']
transaction = globals()['transaction']

AIModel = pool.get('ai.model')
with transaction.set_context(active_test=False):
    models = AIModel.search([('llm_pdf_engine', '=', None)])
    if models:
        AIModel.write(models, {
            'llm_pdf_engine': AIModel.default_llm_pdf_engine(),
            })
transaction.commit()
