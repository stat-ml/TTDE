from dataclasses import dataclass


@dataclass
class Base:
    pass

    def __str__(self):
        return self.__repr__().replace('()', '').replace('(', '_').replace(', ', '_').replace(')', '')
