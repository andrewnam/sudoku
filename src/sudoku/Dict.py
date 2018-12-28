class Dict:

    SUPPORTED_TYPES = {float, int, list, dict, set, str}

    def __init__(self, value_type: type, layers: int=1):
        assert value_type in Dict.SUPPORTED_TYPES
        self.value_type = value_type
        self.layers = layers
        self.dict = {}

    def __setitem__(self, key, item):
        self.dict[key] = item

    def __getitem__(self, key):
        if key not in self.dict:
            if self.layers > 1:
                self.dict[key] = Dict(self.value_type, self.layers-1)
            else:
                if self.value_type is float:
                    self.dict[key] = 0.0
                if self.value_type is int:
                    self.dict[key] = 0
                if self.value_type is list:
                    self.dict[key] = []
                if self.value_type is dict:
                    self.dict[key] = {}
                if self.value_type is set:
                    self.dict[key] = set()
                if self.value_type is str:
                    return ""
        return self.dict[key]

    def __repr__(self):
        return repr(self.dict)

    def __len__(self):
        return len(self.dict)

    def __delitem__(self, key):
        del self.dict[key]

    def clear(self):
        return self.dict.clear()

    def copy(self):
        return self.dict.copy()

    def has_key(self, k):
        return k in self.dict

    def update(self, *args, **kwargs):
        return self.dict.update(*args, **kwargs)

    def keys(self):
        return self.dict.keys()

    def values(self):
        return self.dict.values()

    def items(self):
        return self.dict.items()

    def pop(self, *args):
        return self.dict.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.dict, dict_)

    def __contains__(self, item):
        return item in self.dict

    def __iter__(self):
        return iter(self.dict)

    def __unicode__(self):
        return unicode(repr(self.dict))
