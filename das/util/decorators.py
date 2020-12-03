# -*- coding: utf-8 -*-


def deprecated(func):
    def inner(self, *args, **kwargs):
        print("Deprecated Method: {}".format(func))
        return func(self, *args, **kwargs)
    return inner


def check_model(func):
    def inner(self, *args, **kwargs):
        if self.model is None:
            raise Exception('self.model cannot be None!')
        else:
            return func(self, *args, **kwargs)
    return inner


def check_parameter_space(func):
    def inner(self, *args, **kwargs):
        if self.parameter_space is None:
            self.set_configuration_space()
        return func(self, *args, **kwargs)
    return inner


def check_parameter_space_not_none(func):
    def inner(self, *args, **kwargs):
        if self.parameter_space is None:
            raise Exception('The Parameter Space cannot be None!')
        return func(self, *args, **kwargs)
    return inner
