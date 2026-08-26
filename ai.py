import json
from decimal import Decimal

import requests
from openai import OpenAI
from sql import Table
from trytond import backend
import trytond.config as config_
from trytond.exceptions import UserError
from trytond.i18n import gettext
from trytond.model import (
    DeactivableMixin, ModelSingleton, ModelSQL, ModelView, Unique, fields)
from trytond.pool import Pool, PoolMeta
from trytond.pyson import Bool, Eval
from trytond.transaction import Transaction

OPENAI_KEY = config_.config.get('openai', 'api_key')
OPENAI_ORGANIZATION = config_.config.get('openai', 'organization')
if OPENAI_KEY:
    OPENAI_CLIENT = OpenAI(
        api_key=OPENAI_KEY, organization=OPENAI_ORGANIZATION)
else:
    OPENAI_CLIENT = None

OPENROUTER_KEY = config_.config.get('openrouter', 'api_key')
OPENROUTER_MODELS_URL = 'https://openrouter.ai/api/v1/models'
OPENROUTER_REASONING_PARAMETERS = {'reasoning', 'reasoning_effort'}
OPENROUTER_WEB_SEARCH_PARAMETERS = {'web_search_options'}
TOKENS_PER_MILLION = Decimal(1000000)
MOVED_MODEL_DATA = {
    'view_chat_instruction_form',
    'act_chat_instruction',
    'act_chat_instruction_form',
    'menu_instruction',
    'access_chat_instruction',
    'access_chat_instruction_account_admin',
    'view_openrouter_model_form',
    'view_openrouter_model_tree',
    'cron_sync_openrouter_models',
    'act_openrouter_model',
    'act_openrouter_model_form',
    'act_openrouter_model_tree',
    'access_openrouter_model',
    'access_openrouter_model_admin',
    'menu_openrouter_model',
    'view_ai_model_form',
    'view_ai_model_tree',
    'act_ai_model',
    'act_ai_model_form',
    'act_ai_model_tree',
    'menu_ai_model',
    'access_ai_model',
    'access_ai_model_admin',
    'msg_assistant_unreachable',
    'msg_assistant_authentication_failed',
    'msg_assistant_no_response',
    'msg_openrouter_model_id_unique',
    'msg_openrouter_model_sync_error',
    }
OLD_MODELS = {
    'nantic.ai.model.openrouter',
    'nantic.ai.model',
    'nantic.ai.configuration',
    }
if OPENROUTER_KEY:
    OPENROUTER_CLIENT = OpenAI(
        base_url='https://openrouter.ai/api/v1',
        api_key=OPENROUTER_KEY)
else:
    OPENROUTER_CLIENT = None


def migrate_model_data():
    if not backend.TableHandler.table_exist('ir_model_data'):
        return
    model_data = Table('ir_model_data')
    cursor = Transaction().connection.cursor()
    cursor.execute(*model_data.update(
            [model_data.module], ['ai_model'],
            where=(model_data.module == 'nantic_ai_model')
            | ((model_data.module == 'nantic_connection')
                & model_data.fs_id.in_(tuple(sorted(MOVED_MODEL_DATA))))))


def migrate_models():
    cursor = Transaction().connection.cursor()
    models = [
        ('nantic.ai.model.openrouter', 'ai.model.openrouter'),
        ('nantic.ai.model', 'ai.model'),
        ('nantic.ai.configuration', 'ai.configuration'),
        ]

    remove_old_model_records()

    if backend.TableHandler.table_exist('ir_model'):
        ir_model = Table('ir_model')
        for old_name, new_name in models:
            cursor.execute(*ir_model.update(
                    [ir_model.name, ir_model.module],
                    [new_name, 'ai_model'],
                    where=ir_model.name == old_name))

    if backend.TableHandler.table_exist('ir_model_field'):
        ir_model_field = Table('ir_model_field')
        for old_name, new_name in models:
            cursor.execute(*ir_model_field.update(
                    [ir_model_field.model], [new_name],
                    where=ir_model_field.model == old_name))
            cursor.execute(*ir_model_field.update(
                    [ir_model_field.relation], [new_name],
                    where=ir_model_field.relation == old_name))

    if backend.TableHandler.table_exist('ir_translation'):
        translation = Table('ir_translation')
        cursor.execute(*translation.update(
                [translation.module], ['ai_model'],
                where=translation.module == 'nantic_ai_model'))
        for old_name, new_name in models:
            cursor.execute(*translation.select(
                    translation.id, translation.name,
                    where=translation.name.like(old_name + '%')))
            for translation_id, name in cursor.fetchall():
                cursor.execute(*translation.update(
                        [translation.name], [
                            new_name + name[len(old_name):]],
                        where=translation.id == translation_id))


def remove_old_model_records():
    tables = {
        'ir_action_act_window',
        'ir_action_act_window_view',
        'ir_model_access',
        'ir_model_field',
        'ir_model_data',
        'ir_ui_view',
        }
    if not all(backend.TableHandler.table_exist(table) for table in tables):
        return

    cursor = Transaction().connection.cursor()
    action = Table('ir_action_act_window')
    cursor.execute(*action.select(
            action.id,
            where=(action.res_model.in_(tuple(OLD_MODELS)))
            | (action.context_model.in_(tuple(OLD_MODELS)))))
    action_ids = [action_id for action_id, in cursor]

    action_view = Table('ir_action_act_window_view')
    action_view_ids = []
    if action_ids:
        cursor.execute(*action_view.select(
                action_view.id,
                where=action_view.act_window.in_(action_ids)))
        action_view_ids = [action_view_id for action_view_id, in cursor]

    access = Table('ir_model_access')
    cursor.execute(*access.select(
            access.id, where=access.model.in_(tuple(OLD_MODELS))))
    access_ids = [access_id for access_id, in cursor]

    view = Table('ir_ui_view')
    cursor.execute(*view.select(
            view.id, where=view.model.in_(tuple(OLD_MODELS))))
    view_ids = [view_id for view_id, in cursor]

    model_data = Table('ir_model_data')
    where = None
    if action_ids:
        where = ((model_data.model == 'ir.action.act_window')
            & model_data.db_id.in_(action_ids))
    if action_view_ids:
        condition = ((model_data.model == 'ir.action.act_window.view')
            & model_data.db_id.in_(action_view_ids))
        where = condition if where is None else where | condition
    if access_ids:
        condition = ((model_data.model == 'ir.model.access')
            & model_data.db_id.in_(access_ids))
        where = condition if where is None else where | condition
    if view_ids:
        condition = ((model_data.model == 'ir.ui.view')
            & model_data.db_id.in_(view_ids))
        where = condition if where is None else where | condition
    if where is not None:
        cursor.execute(*model_data.delete(where=where))
    pool = Pool()
    ActionWindow = pool.get('ir.action.act_window')
    if action_ids:
        ActionWindow.delete(ActionWindow.browse(action_ids))
    if access_ids:
        cursor.execute(*access.delete(where=access.id.in_(access_ids)))
    if view_ids:
        cursor.execute(*view.delete(where=view.id.in_(view_ids)))
    model_field = Table('ir_model_field')
    cursor.execute(*model_field.delete(
            where=(model_field.model.in_(tuple(OLD_MODELS)))
            | (model_field.relation.in_(tuple(OLD_MODELS)))))
    model_data_model = pool.get('ir.model.data')
    model_data_model._get_id_cache.clear()
    model_data_model._has_model_cache.clear()
    pool.get('ir.model.access')._get_access_cache.clear()


def migrate_model_fields(Model):
    if not backend.TableHandler.table_exist('ir_model_field'):
        return
    ir_model_field = Table('ir_model_field')
    cursor = Transaction().connection.cursor()
    cursor.execute(*ir_model_field.update(
            [ir_model_field.module], ['ai_model'],
            where=(ir_model_field.model == Model.__name__)
            & ir_model_field.name.in_(tuple(sorted(Model._fields)))))


class Cron(metaclass=PoolMeta):
    __name__ = 'ir.cron'

    @classmethod
    def __setup__(cls):
        super().__setup__()
        cls.method.selection.extend([
                ('ai.model.openrouter|cron_sync',
                    'Sync OpenRouter Models'),
                ])


def _join(values):
    if not values:
        return None
    return ', '.join(str(value) for value in values if value)


def _as_int(value):
    if value in (None, ''):
        return None
    return int(value)


def _price_per_million_tokens(value):
    if value in (None, ''):
        return value
    return format(Decimal(value) * TOKENS_PER_MILLION, 'f')


def _split_parameters(value):
    if not value:
        return set()
    return {
        parameter.strip()
        for parameter in value.split(',')
        if parameter.strip()
        }


class OpenRouterModel(DeactivableMixin, ModelSQL, ModelView):
    'OpenRouter Model'
    __name__ = 'ai.model.openrouter'

    openrouter_id = fields.Char('OpenRouter ID', required=True, readonly=True)
    name = fields.Char('Name', required=True, readonly=True)
    description = fields.Text('Description', readonly=True)
    created = fields.Integer('Created', readonly=True)
    context_length = fields.Integer('Context Length', readonly=True)
    prompt_price = fields.Char('Prompt Price', readonly=True)
    completion_price = fields.Char('Completion Price', readonly=True)
    request_price = fields.Char('Request Price', readonly=True)
    image_price = fields.Char('Image Price', readonly=True)
    web_search_price = fields.Char('Web Search Price', readonly=True)
    internal_reasoning_price = fields.Char(
        'Internal Reasoning Price', readonly=True)
    input_cache_read_price = fields.Char(
        'Input Cache Read Price', readonly=True)
    input_cache_write_price = fields.Char(
        'Input Cache Write Price', readonly=True)
    modality = fields.Char('Modality', readonly=True)
    input_modalities = fields.Char('Input Modalities', readonly=True)
    output_modalities = fields.Char('Output Modalities', readonly=True)
    tokenizer = fields.Char('Tokenizer', readonly=True)
    instruct_type = fields.Char('Instruct Type', readonly=True)
    supported_parameters = fields.Text('Supported Parameters', readonly=True)
    top_provider_context_length = fields.Integer(
        'Top Provider Context Length', readonly=True)
    top_provider_max_completion_tokens = fields.Integer(
        'Top Provider Max Completion Tokens', readonly=True)
    top_provider_is_moderated = fields.Boolean(
        'Top Provider Is Moderated', readonly=True)
    per_request_limits = fields.Text('Per Request Limits', readonly=True)
    raw_data = fields.Text('Raw Data', readonly=True)

    @classmethod
    def __setup__(cls):
        super().__setup__()
        table = cls.__table__()
        cls._sql_constraints += [(
                'openrouter_id_unique',
                Unique(table, table.openrouter_id),
                'ai_model.msg_openrouter_model_id_unique')]

    @classmethod
    def __register__(cls, module_name):
        cursor = Transaction().connection.cursor()
        old_table = 'nantic_ai_model_openrouter'
        if (backend.TableHandler.table_exist(old_table)
                and not backend.TableHandler.table_exist(cls._table)):
            backend.TableHandler.table_rename(old_table, cls._table)
        migrate_models()
        migrate_model_fields(cls)
        migrate_model_data()
        super().__register__(module_name)
        table = cls.__table__()
        cursor.execute(*table.update(
                [table.active], [True], where=table.active == None))

    def get_rec_name(self, name):
        return '%s (%s)' % (self.name, self.openrouter_id)

    @classmethod
    def search_rec_name(cls, name, clause):
        return [
            'OR',
            ('name',) + tuple(clause[1:]),
            ('openrouter_id',) + tuple(clause[1:]),
            ]

    def get_supported_parameters(self):
        parameters = _split_parameters(self.supported_parameters)
        if self.raw_data:
            try:
                raw_data = json.loads(self.raw_data)
            except ValueError:
                raw_data = {}
            parameters.update(raw_data.get('supported_parameters') or [])
        return parameters

    def supports_any_parameter(self, parameters):
        return bool(self.get_supported_parameters() & set(parameters))

    @classmethod
    def _openrouter_headers(cls):
        headers = {'Accept': 'application/json'}
        if OPENROUTER_KEY:
            headers['Authorization'] = 'Bearer %s' % OPENROUTER_KEY
        return headers

    @classmethod
    def _fetch_openrouter_models(cls):
        try:
            response = requests.get(
                OPENROUTER_MODELS_URL, headers=cls._openrouter_headers(),
                timeout=30)
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.RequestException as exception:
            raise UserError(gettext(
                    'ai_model.msg_openrouter_model_sync_error',
                    error=str(exception)))
        except ValueError as exception:
            raise UserError(gettext(
                    'ai_model.msg_openrouter_model_sync_error',
                    error=str(exception)))
        return payload.get('data', [])

    @classmethod
    def _values_from_payload(cls, data):
        architecture = data.get('architecture') or {}
        pricing = data.get('pricing') or {}
        top_provider = data.get('top_provider') or {}
        return {
            'openrouter_id': data.get('id'),
            'name': data.get('name') or data.get('id'),
            'description': data.get('description'),
            'created': _as_int(data.get('created')),
            'context_length': _as_int(data.get('context_length')),
            'prompt_price': _price_per_million_tokens(pricing.get('prompt')),
            'completion_price': _price_per_million_tokens(
                pricing.get('completion')),
            'request_price': pricing.get('request'),
            'image_price': pricing.get('image'),
            'web_search_price': pricing.get('web_search'),
            'internal_reasoning_price': _price_per_million_tokens(
                pricing.get('internal_reasoning')),
            'input_cache_read_price': _price_per_million_tokens(
                pricing.get('input_cache_read')),
            'input_cache_write_price': _price_per_million_tokens(
                pricing.get('input_cache_write')),
            'modality': architecture.get('modality'),
            'input_modalities': _join(architecture.get('input_modalities')),
            'output_modalities': _join(architecture.get('output_modalities')),
            'tokenizer': architecture.get('tokenizer'),
            'instruct_type': architecture.get('instruct_type'),
            'supported_parameters': _join(data.get('supported_parameters')),
            'top_provider_context_length': _as_int(
                top_provider.get('context_length')),
            'top_provider_max_completion_tokens': _as_int(
                top_provider.get('max_completion_tokens')),
            'top_provider_is_moderated': top_provider.get('is_moderated'),
            'per_request_limits': json.dumps(
                data.get('per_request_limits'), indent=2, sort_keys=True),
            'raw_data': json.dumps(data, indent=2, sort_keys=True),
            'active': True,
            }

    @classmethod
    def sync_models(cls):
        models = cls._fetch_openrouter_models()
        existing = {
            model.openrouter_id: model
            for model in cls.search([], order=[])
            }
        seen = set()
        to_create = []

        for model in models:
            if not model.get('id'):
                continue
            values = cls._values_from_payload(model)
            seen.add(values['openrouter_id'])
            existing_model = existing.get(values['openrouter_id'])
            if existing_model:
                cls.write([existing_model], values)
            else:
                to_create.append(values)
        if to_create:
            cls.create(to_create)

        outdated = [
            model for openrouter_id, model in existing.items()
            if openrouter_id not in seen and model.active]
        if outdated:
            cls.write(outdated, {'active': False})

    @classmethod
    def cron_sync(cls):
        cls.sync_models()


class AIModel(DeactivableMixin, ModelSQL, ModelView):
    'AI Model'
    __name__ = 'ai.model'

    name = fields.Char('Name', required=True)
    type = fields.Selection([
            ('llm', 'LLM'),
            ('embedding', 'Embedding'),
            ], 'Type')
    provider = fields.Selection([
            ('openrouter', 'OpenRouter'),
            ('openai', 'OpenAI'),
            ], 'Provider', required=True)
    openrouter_model = fields.Many2One(
        'ai.model.openrouter', 'OpenRouter Model',
        domain=[('active', '=', True)],
        states={
            'invisible': Eval('provider') != 'openrouter',
            },
        depends=['provider'])
    model_name = fields.Char('Model Name', required=True)
    supports_reasoning = fields.Function(fields.Boolean('Supports Reasoning'),
        'on_change_with_supports_reasoning')
    supports_web_search = fields.Function(fields.Boolean(
            'Supports Web Search'), 'on_change_with_supports_web_search')
    reasoning = fields.Selection([
            (None, 'None'),
            ('low', 'Low'),
            ('medium', 'Medium'),
            ('high', 'High'),
            ('xhigh', 'XHigh'),
            ], 'Reasoning', states={
                'invisible': (
                    (Eval('type') != 'llm')
                    | (Bool(Eval('openrouter_model'))
                        & ~Eval('supports_reasoning'))),
                }, depends=[
                    'type', 'openrouter_model', 'supports_reasoning'])
    allow_web_search = fields.Boolean('Allow Web Search', states={
            'invisible': (
                (Eval('type') != 'llm')
                | (Bool(Eval('openrouter_model'))
                    & ~Eval('supports_web_search'))),
            }, depends=[
                'type', 'openrouter_model', 'supports_web_search'])

    @classmethod
    def __register__(cls, module_name):
        cursor = Transaction().connection.cursor()
        old_table = 'nantic_ai_model'
        if (backend.TableHandler.table_exist(old_table)
                and not backend.TableHandler.table_exist(cls._table)):
            backend.TableHandler.table_rename(old_table, cls._table)
        migrate_models()
        migrate_model_fields(cls)
        super().__register__(module_name)
        table = cls.__table__()
        cursor.execute(*table.update(
                [table.type], ['llm'], where=table.type == None))

    @staticmethod
    def default_provider():
        return 'openrouter'

    @staticmethod
    def default_type():
        return 'llm'

    @fields.depends('openrouter_model')
    def on_change_openrouter_model(self):
        if self.openrouter_model:
            self.model_name = self.openrouter_model.openrouter_id
            if not getattr(self, 'name', None):
                self.name = self.openrouter_model.name

    @fields.depends(
        'openrouter_model', '_parent_openrouter_model.supported_parameters',
        '_parent_openrouter_model.raw_data')
    def on_change_with_supports_reasoning(self, name=None):
        if not self.openrouter_model:
            return True
        return self.openrouter_model.supports_any_parameter(
            OPENROUTER_REASONING_PARAMETERS)

    @fields.depends(
        'openrouter_model', '_parent_openrouter_model.supported_parameters',
        '_parent_openrouter_model.raw_data')
    def on_change_with_supports_web_search(self, name=None):
        if not self.openrouter_model:
            return True
        return self.openrouter_model.supports_any_parameter(
            OPENROUTER_WEB_SEARCH_PARAMETERS)

    @fields.depends('provider', 'openrouter_model')
    def on_change_provider(self):
        if self.provider != 'openrouter':
            self.openrouter_model = None

    def get_client(self):
        if self.provider == 'openai':
            return OPENAI_CLIENT
        return OPENROUTER_CLIENT

    @classmethod
    def get_default_llm_client(cls):
        return OPENROUTER_CLIENT or OPENAI_CLIENT

    @classmethod
    def get_default_embedding_client(cls):
        return OPENAI_CLIENT


class AIConfiguration(ModelSingleton, ModelSQL, ModelView):
    'AI Configuration'
    __name__ = 'ai.configuration'

    default_llm = fields.Many2One('ai.model', 'Default LLM',
        domain=[('type', '=', 'llm')], ondelete='RESTRICT')
    default_embedding_model = fields.Many2One('ai.model',
        'Default Embedding', domain=[('type', '=', 'embedding')],
        ondelete='RESTRICT')

    @classmethod
    def __register__(cls, module_name):
        for old_table in (
                'nantic_ai_configuration', 'nantic_chat_instruction'):
            if (backend.TableHandler.table_exist(old_table)
                    and not backend.TableHandler.table_exist(cls._table)):
                backend.TableHandler.table_rename(old_table, cls._table)

        migrate_models()
        migrate_model_fields(cls)

        handler = cls.__table_handler__(module_name)
        if (handler.column_exist('default_embedding')
                and not handler.column_exist('default_embedding_model')):
            handler.column_rename(
                'default_embedding', 'default_embedding_model')
        super().__register__(module_name)

    def get_default_llm_client(self):
        if self.default_llm:
            return self.default_llm.get_client()
        return AIModel.get_default_llm_client()

    def get_default_embedding_client(self):
        if self.default_embedding_model:
            return self.default_embedding_model.get_client()
        return AIModel.get_default_embedding_client()
