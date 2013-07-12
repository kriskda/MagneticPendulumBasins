

class CommonFunctions(object):
    
    @staticmethod
    def array_to_float_carray(array):
        list_elements = ', '.join(map(lambda x: str(x) + "f", array))
        
        return "{" + list_elements  + "}" 