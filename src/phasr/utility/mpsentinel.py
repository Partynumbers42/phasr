class MPSentinel(object):
    
    _is_master = False
    
    @classmethod
    def As_master(cls):
        cls._is_master = True

    @classmethod
    def Is_master(cls):
        return cls._is_master
