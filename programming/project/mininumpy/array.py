class Array:
    """Array data to emulate numpy class"""

    stored_data: list[any]
    data: memoryview
    shape: tuple[int]
    ndim: int
    size: int

    def __init__(self, input_list: list[int | float]):
        # get data dimension
        self.shape = self._get_list_shape(input_list)[0]
        self.ndim = len(self.shape)
        self.size = self._multiply_int_list(self.shape)

        # actually store data
        self.stored_data = input_list
        # TODO: data in memoryview still missing

    @staticmethod
    def _multiply_int_list(lst: int) -> int:
        accum = 1
        for elem in lst:
            accum *= elem
        return accum

    @staticmethod
    def _is_same_or_castable(
        type_1: int | float | list | None,
        type_2: int | float | list | None,
    ) -> bool:
        if type_1 == type_2:
            return True
        if {type_1, type_2} == {int, float}:
            return True
        return False

    @classmethod
    def _get_list_shape(
        cls, test_list: int | float | list,
    ) -> tuple[tuple[int], int | float | None]:
        # check if the list has consistent dimensions (no sublists of different)
        # length
        if isinstance(test_list, list):
            length = len(test_list)
            if length == 0:
                return (0, None)

            shapes = [cls._get_list_shape(elem) for elem in test_list]
            shape_1 = shapes[0][0]
            type_1 = shapes[0][1]
            for shape in shapes[1:]:
                if shape[0] != shape_1:
                    raise ValueError("Inconsistent shape between sublists")
                if not cls._is_same_or_castable(type_1, shape[1]):
                    raise ValueError("Inconsistent typing between sublists' elements")
            return (length, *shape_1), type_1

        if isinstance(test_list, int) or isinstance(test_list, float):
            return (), type(test_list)

        else:
            return ValueError("Cannot get the shape of non-list, non-nummber type")

    def reshape(self, new_shape: tuple[int]):
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
            else:  # should be a list instace
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
    def eye(cls, n: int) -> Array:
        array = cls.zeros((n, n))
        for i in range(n):
            array.stored_data[i][i] = 1

        return array

    @staticmethod
    def _check_range(start: float, stop: float) -> None:
        if start > stop:
            raise RuntimeError("Invalid range, start > stop")

    @staticmethod
    def _check_postive(number: float, name: str) -> None:
        if number <= 0:
            raise RuntimeError(f"Invalid value ( <=0 ) for {name}")

    @classmethod
    def arange(cls, start: float, stop: float, step: float) -> Array:
        # Array of values from [start,stop), with difference of
        # step in between each pair

        # sanitize input
        cls._check_range(start, stop)
        cls._check_positive(step, "step")

        base_list = []
        idx = 0
        while start + idx * step < stop:
            base_list.append(start + step * idx)
            idx += 1

        return cls.array(base_list)

    @classmethod
    def linspace(cls, start: float, stop: float, num: int) -> Array:
        # evenly num-spaced values in the interval [start,stop)

        # sanitize input
        cls._check_range(start, stop)
        cls._check_positive(num, "num")

        diff = (stop - start) / num
        base_list = []
        for idx in range(num):
            base_list.append(start + diff * idx)

        return cls.array(base_list)
