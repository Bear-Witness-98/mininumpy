class Array:
    """Array data to emulate numpy class"""

    stored_data: list[any]
    data: memoryview
    shape: tuple[int]
    ndim: int
    size: int

    def __init__(self, sanitized_list:list[str | int | float]):
        self.stored_data = sanitized_list
        # fix other stuff

    # Here the output should be an instance of this same class.
    # But in the typehints you just put the class, not the instance
    def reshape(self, new_shape:tuple[int]):
        pass

    def transpose(self):
        pass

    def __str__():
        pass


    
class MiniNumPy:

    @staticmethod
    def array(list_or_nested_list: list) -> Array:
        # Sanitize array
        return Array(list_or_nested_list)
    

    @classmethod
    def _singular_value_array(cls, shape: tuple[int], value: int | float) -> Array:
        
        # construct a mega list with zeros
        # sanitize input
        ndim = len(shape)
        size = 1
        for dim_length in shape:
            size *= dim_length
        
        current_value = value
        for dim in shape[::-1]:
            if isinstance(current_value, int) or isinstance(current_value, float):
                current_list_level = [current_value for _ in range(dim)]
            else: # should be a list instace
                current_list_level = [current_value.copy() for _ in range(dim)]
            current_value = current_list_level

        return cls.array(current_list_level)
    
    @classmethod
    def zeros(cls, shape: tuple[int]) -> Array:
        return cls._singular_value_array(shape, 0)

    @classmethod
    def ones(cls, shape: tuple[int]) -> Array:
        return cls._singular_value_array(shape, 1)

    @classmethod
    def eye(cls, n:int) -> Array:
        array = cls.zeros((n,n))
        for i in range(n):
            array.stored_data[i][i] = 1

        return array

    @classmethod
    def arange(cls, start:float, stop:float, step:float) -> Array:
        # evenly spaced values in the interval [start,stop)
        # sanitize input (zero step, stop > start, etc).
        # this has some issues, fix
        n_values = int((stop - start) / step)
        base_list = []
        for idx in range(n_values):
            base_list.append(start + step*idx)

        return cls.array(base_list)

    @classmethod
    def linspace(cls, start:float, stop:float, num:int) -> Array:
        
        diff = int((stop - start) / num)
        base_list = []
        for idx in range(num):
            base_list.append(start + diff*idx)

        return cls.array(base_list)





