import logging
import time

from openai import APIStatusError, AuthenticationError, PermissionDeniedError
from trytond.i18n import gettext
from trytond.pool import Pool
from unidecode import unidecode

from .ai import OPENAI_CLIENT, OPENROUTER_CLIENT

logger = logging.getLogger(__name__)

MODEL_LOW = 'openai/gpt-5.6-terra'


def get_default_llm_client():
    Configuration = Pool().get('ai.configuration')
    return Configuration(1).get_default_llm_client()


def get_completion(messages, model=MODEL_LOW, tools=None, tool_choice=None,
        user=None, client=None):
    provider = None
    model_name = model
    kwargs = {}
    client = client or get_default_llm_client()

    if hasattr(model, 'provider') and hasattr(model, 'model_name'):
        provider = model.provider
        model_name = model.model_name
        client = model.get_client()
        if provider == 'openai':
            if model.reasoning:
                kwargs['reasoning_effort'] = model.reasoning
            if model.allow_web_search:
                kwargs['web_search_options'] = {}
    else:
        Configuration = Pool().get('ai.configuration')
        configuration = Configuration(1)
        if configuration.default_llm:
            provider = configuration.default_llm.provider
        elif client is OPENAI_CLIENT:
            provider = 'openai'
        elif client is OPENROUTER_CLIENT:
            provider = 'openrouter'
    if not client:
        if provider:
            logger.error('AI provider "%s" is not configured.', provider)
        else:
            logger.error('AI service is not configured.')
        return 'AI server is not configured', True
    if user:
        user = unidecode(user)
    else:
        user = f'{Pool().database_name}:root'

    for retry in range(3):
        try:
            started_at = time.monotonic()
            logger.info('Calling...')
            response = client.chat.completions.create(
                user=user,
                model=model_name,
                messages=messages,
                tools=tools or [],
                tool_choice=tool_choice or 'auto',
                parallel_tool_calls=False,
                **kwargs)
            elapsed_ms = round((time.monotonic() - started_at) * 1000, 3)
            logger.info(
                'AI completion API call finished in %s ms '
                '(provider=%s, model=%s).',
                elapsed_ms, provider or 'default', model_name)
            logger.warning('AI completion response: %s', response)
            if not response.choices:
                logger.warning('AI completion API call returned no choices.')
                if retry < 2:
                    time.sleep(2 ** retry)
                    logger.info(
                        'Retrying completion (attempt %d of 3)...', retry + 2)
                    continue
                return gettext(
                    'ai_model.msg_assistant_no_response'), True

            return response, False
        except APIStatusError:
            logger.warning('OpenAI status error.', exc_info=True)
            return gettext(
                'ai_model.msg_assistant_authentication_failed'), True
        except PermissionDeniedError:
            logger.warning('OpenAI permission error.', exc_info=True)
            return gettext(
                'ai_model.msg_assistant_authentication_failed'), True
        except AuthenticationError:
            logger.warning('OpenAI authentication error.', exc_info=True)
            return gettext(
                'ai_model.msg_assistant_authentication_failed'), True
        except Exception as error:
            logger.exception('While getting completion.')
            if retry < 2:
                time.sleep(2 ** retry)
                logger.info(
                    'Retrying completion (attempt %d of 3)...', retry + 2)
                continue
            if getattr(error, 'code', None) == 502:
                return gettext(
                    'ai_model.msg_assistant_unreachable'), True
            return gettext('ai_model.msg_internal_server_error'), True
