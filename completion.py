import datetime
import logging
import time
from decimal import Decimal

from openai import APIStatusError, AuthenticationError, PermissionDeniedError
from trytond.i18n import gettext
from trytond.pool import Pool
from trytond.transaction import without_check_access
from unidecode import unidecode

from .ai import DEFAULT_LLM_MODEL

logger = logging.getLogger(__name__)

MODEL_LOW = DEFAULT_LLM_MODEL


def _get_attr_or_key(data, name, default=None):
    if data is None:
        return default
    if isinstance(data, dict):
        return data.get(name, default)
    return getattr(data, name, default)


def _get_usage_tokens(usage):
    input_tokens = (_get_attr_or_key(usage, 'prompt_tokens')
        or _get_attr_or_key(usage, 'input_tokens') or 0)
    output_tokens = (_get_attr_or_key(usage, 'completion_tokens')
        or _get_attr_or_key(usage, 'output_tokens') or 0)
    details = (_get_attr_or_key(usage, 'prompt_tokens_details')
        or _get_attr_or_key(usage, 'input_tokens_details'))
    cached_tokens = _get_attr_or_key(details, 'cached_tokens', 0) or 0
    return input_tokens, cached_tokens, output_tokens


def _compute_openai_cost(model, usage):
    input_tokens, cached_tokens, output_tokens = _get_usage_tokens(usage)
    model_name = model.model_name.split('/')[-1]
    prices = {
        'gpt-5': ('1.25', '0.125', '10'),
        'gpt-5.2': ('1.75', '0.175', '14'),
        'gpt-5.4': ('2.50', '0.25', '15'),
        'gpt-5.4-mini': ('0.75', '0.075', '4.50'),
        'gpt-5.4-nano': ('0.20', '0.02', '1.25'),
        'gpt-5-mini': ('0.25', '0.025', '2'),
        'gpt-5-nano': ('0.05', '0.005', '0.4'),
        }
    matched_model = next((name for name in sorted(
                prices, key=len, reverse=True)
            if model_name == name or model_name.startswith(name + '-')),
        None)
    if matched_model is None:
        return None
    prompt_price, cached_price, completion_price = prices[matched_model]
    uncached_tokens = max(0, input_tokens - cached_tokens)
    return (
        Decimal(uncached_tokens) * Decimal(prompt_price)
        + Decimal(cached_tokens) * Decimal(cached_price)
        + Decimal(output_tokens) * Decimal(completion_price)
        ) / Decimal('1000000')


def get_completion_cost(response, model):
    usage = _get_attr_or_key(response, 'usage')
    amount = None
    currency = None
    if usage is not None:
        amount = _get_attr_or_key(usage, 'cost')
        currency = (_get_attr_or_key(usage, 'cost_currency')
            or _get_attr_or_key(usage, 'currency'))
        if (model.provider == 'openrouter'
                and _get_attr_or_key(usage, 'is_byok', False)):
            details = _get_attr_or_key(usage, 'cost_details')
            upstream_amount = _get_attr_or_key(
                details, 'upstream_inference_cost')
            if upstream_amount is not None:
                amount = (Decimal(str(amount or 0))
                    + Decimal(str(upstream_amount)))
    if amount is None:
        amount = _get_attr_or_key(response, 'cost')
    if currency is None:
        currency = (_get_attr_or_key(response, 'cost_currency')
            or _get_attr_or_key(response, 'currency'))
    if (amount is None and usage is not None
            and model.provider == 'openai'):
        amount = _compute_openai_cost(model, usage)
    if amount is not None:
        amount = Decimal(str(amount))
    if currency is None and model.provider in {'openai', 'openrouter'}:
        currency = 'USD'
    return amount, currency


def register_cost(response, model, origin, duration):
    if not getattr(origin, 'id', None):
        raise ValueError('The cost origin must be a saved record')
    Cost = Pool().get('ai.model.cost')
    usage = _get_attr_or_key(response, 'usage')
    input_tokens, cached_tokens, output_tokens = _get_usage_tokens(usage)
    amount, currency = get_completion_cost(response, model)
    with without_check_access():
        Cost.create([{
                    'origin': str(origin),
                    'model': model.id,
                    'input_tokens': input_tokens,
                    'cached_input_tokens': cached_tokens,
                    'output_tokens': output_tokens,
                    'cost': amount,
                    'currency': currency,
                    'duration': duration,
                    }])


def get_completion(model, messages, origin, tools=None, tool_choice=None,
        user=None, client=None, response_format=None, extra_body=None,
        store=None, max_tokens=None):
    if not getattr(origin, 'id', None):
        raise ValueError('get_completion origin must be a saved record')
    if model.type != 'llm':
        raise ValueError('get_completion requires an LLM model')
    client = client or model.get_client()
    if not client:
        logger.error('AI provider "%s" is not configured.', model.provider)
        return 'AI server is not configured', True
    if user:
        user = unidecode(user)
    else:
        user = f'{Pool().database_name}:root'

    model_kwargs = {}
    if model.provider == 'openai':
        if model.reasoning:
            model_kwargs['reasoning_effort'] = model.reasoning
        if model.allow_web_search:
            model_kwargs['web_search_options'] = {}

    if model.provider == 'openrouter' and model.llm_pdf_engine:
        has_file = any(
            isinstance(message, dict)
            and isinstance(message.get('content'), list)
            and any(
                isinstance(part, dict) and part.get('type') == 'file'
                for part in message['content'])
            for message in messages or [])
        if has_file:
            extra_body = dict(extra_body or {})
            plugins = list(extra_body.get('plugins') or [])
            if not any(
                    isinstance(plugin, dict)
                    and plugin.get('id') == 'file-parser'
                    for plugin in plugins):
                plugins.append({
                        'id': 'file-parser',
                        'pdf': {'engine': model.llm_pdf_engine},
                        })
                extra_body['plugins'] = plugins

    for retry in range(3):
        try:
            started_at = time.monotonic()
            logger.info('Calling...')
            request = {
                'user': user,
                'model': model.model_name,
                'messages': messages,
                'tools': tools or [],
                'tool_choice': tool_choice or 'auto',
                'parallel_tool_calls': False,
                }
            request.update(model_kwargs)
            if response_format is not None:
                request['response_format'] = response_format
            if extra_body is not None:
                request['extra_body'] = extra_body
            if store is not None:
                request['store'] = store
            if max_tokens is not None:
                request['max_tokens'] = max_tokens
            response = client.chat.completions.create(**request)
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

        duration = datetime.timedelta(seconds=time.monotonic() - started_at)
        logger.info(
            'AI completion API call finished in %s '
            '(provider=%s, model=%s).',
            duration, model.provider, model.model_name)
        logger.warning('AI completion response: %s', response)
        register_cost(response, model, origin, duration)
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


def get_embeddings(model, input_, origin, client=None,
        encoding_format='float', dimensions=None):
    if not getattr(origin, 'id', None):
        raise ValueError('get_embeddings origin must be a saved record')
    if model.type != 'embedding':
        raise ValueError('get_embeddings requires an embedding model')
    client = client or model.get_client()
    if not client:
        raise RuntimeError('OpenRouter is not configured')

    request = {
        'model': model.model_name,
        'input': input_,
        'encoding_format': encoding_format,
        }
    if dimensions is not None:
        request['dimensions'] = dimensions
    started_at = time.monotonic()
    response = client.embeddings.create(**request)
    duration = datetime.timedelta(seconds=time.monotonic() - started_at)
    logger.info(
        'AI embeddings API call finished in %s (model=%s).',
        duration, model.model_name)
    register_cost(response, model, origin, duration)
    return response
