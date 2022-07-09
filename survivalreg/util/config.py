from typing import Any
import os
import json


class Parser(object):
    def __init__(self,
                 name: str, default: Any = None,
                 type_: callable = None, help_info: str = None):
        self.name = name
        self.default = default
        self.type_ = type_
        self.help = help_info

    def __set__(self, instance, value):
        setattr(self, 'value', value)

    def __get__(self, instance, owner):
        if hasattr(self, 'value'):
            return getattr(self, 'value')
        v = os.environ.get(self.name, self.default)
        if self.type_ is not None and self.name in os.environ:
            v = self.type_(v)
        setattr(self, 'value', v)
        return v

    def __call__(self, s):
        if self.type_ is not None:
            return self.type_(s)
        return s

    def __str__(self):
        return f'<{self.name}: {self.help} default:{self.default}>'

    def __repr__(self) -> str:
        return self.__str__()


class Config(object):
    def __init__(self):
        pass

    @property
    def value_dict(self):
        return self._search_cfg_recursively(self.__class__)

    @staticmethod
    def _search_cfg_recursively(root):
        vals = dict()
        for base in root.__bases__:
            vals.update(Config._search_cfg_recursively(base))
        for k, v in root.__dict__.items():
            if isinstance(v, Parser):
                vals[k] = v.__get__(None, None)
        return vals

    def __repr__(self):
        return f'<{self.__class__.__name__}: {json.dumps(self.value_dict)}>'

    def sample_cfg(self):
        for k, v in self.__class__.__dict__.items():
            if isinstance(v, Parser):
                print(f'{k}={v.default} ', end='')
        print()
