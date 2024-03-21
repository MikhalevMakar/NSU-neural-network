from enum import Enum, auto


class Brand(Enum):
    ACER = auto()
    LENOVO = auto()
    HP = auto()
    ASUS = auto()
    DELL = auto()


class EnumNotFoundException(Exception):
    pass


class EnumUtils:
    @staticmethod
    def parse(enum_type, name):
        try:
            return enum_type[name.upper()]
        except KeyError:
            raise EnumNotFoundException(enum_type, name)
