
class AbstractRelation(object):

    def __init__(self):
        pass

    def judge(self, params):
        """

        :param params: dict type
        :return:
        """
        raise NotImplementedError

    def adjust_params(self, params):
        raise NotImplementedError


class GreaterThanRelation(AbstractRelation):

    def __init__(self,
                 key1,
                 key2):
        super(GreaterThanRelation, self).__init__()
        self.key1 = key1
        self.key2 = key2

    def judge(self, params):
        """

        :param params: a tuple containing two parameters
        :return: bool value indicates whether params are good param pairs
        """

        if not isinstance(params, dict):
            raise Exception("parameters should be a dict!")

        try:
            if params[self.key1] < params[self.key2]:
                return False
        except Exception as exc:
            print(exc)
            return False

        return True

    def adjust_params(self, params):

        params[self.key1], params[self.key2] = params[self.key2], params[self.key1]


class LessThanRelation(AbstractRelation):

    def __init__(self,
                 key1,
                 key2
                 ):
        super(LessThanRelation, self).__init__()
        self.key1 = key1
        self.key2 = key2

    def judge(self, params):

        try:
            if params[self.key1] > params[self.key2]:
                return False
        except Exception as exc:
            print(exc)
            return False

    def adjust_params(self, params):

        params[self.key1], params[self.key2] = params[self.key2], params[self.key1]


class EqualRelation(AbstractRelation):

    def __init__(self,
                 key1,
                 key2
                 ):
        super(EqualRelation, self).__init__()
        self.key1 = key1
        self.key2 = key2

    def judge(self, params):

        try:
            if params[self.key2] != params[self.key1]:
                return False
        except Exception as exc:
            print(exc)
            return False

        return True

    def adjust_params(self, params):

        params[self.key2] = params[self.key1]


class NotEqualRelation(AbstractRelation):

    def __init__(self,
                 key1,
                 key2):
        super(NotEqualRelation, self).__init__()
        self.key1 = key1
        self.key2 = key2

    def judge(self, params):

        if params[self.key1] == params[self.key2]:
            return False

        return True

    def adjust_params(self, params):
        pass


class ConditionRelation(AbstractRelation):

    def __init__(self,
                 param1,
                 param2):
        """

        :param param1: tuple type
        :param param2: tuple type
        """
        super(ConditionRelation, self).__init__()
        self._param1 = param1
        self._param2 = param2

    def judge(self, params):
        if params[self._param1[0]] != self._param1[1]:
            return True
        elif params[self._param2[0]] == self._param2[1]:
            return True
        else:
            return False

    def set_name(self, name1, name2):
        self._param1 = name1
        self._param2 = name2

    def get_name(self):
        return self._param1, self._param2

    def adjust_params(self, params):
        pass


class NotRelation(AbstractRelation):

    def __init__(self,
                 key,
                 value):
        super(NotRelation, self).__init__()
        self._key = key
        self._value = value

    def judge(self, params):
        if params[self._key] == self._value:
            return False
        return True

    def adjust_params(self, params):
        pass
