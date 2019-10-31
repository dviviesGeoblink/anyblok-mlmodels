from anyblok.blok import Blok
from logging import getLogger
logger = getLogger(__name__)


class AddressBlok(Blok):
    """Machine Learning blok
    """
    version = "0.1.0"
    author = "Denis Viviès"
    required = []

    @classmethod
    def import_declaration_module(cls):
        from . import mlmdodels # noqa

    @classmethod
    def reload_declaration_module(cls, reload):
        from . import address
        reload(address)