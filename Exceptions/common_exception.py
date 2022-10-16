class ParamTypeError(Exception):
    def __init__(self, type1, type2):
        Exception.__init__(self, "Expected {}, now is {}.".format(type1.upper(), type2.upper()))
