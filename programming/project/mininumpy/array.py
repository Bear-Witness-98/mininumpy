class Array:
	"""Array data to emulate numpy class"""

	data: memoryview
	data_list = list
	data_array: list[int | float | list]
	shape: tuple[int]
	dtype: type[int] | type[float] | type[None]
	ndim: int
	size: int

	# TODO: get the memoryview thing correctly
	# TODO: check the type consitency, if casting is necessary
	def __init__(self, input_list: list[int | float]):
		# converts list into Array type
		self.shape, self.dtype = self._get_list_shape(input_list)
		self.ndim = len(self.shape)
		self.size = self._multiply_int_list(self.shape)

		# actually store data
		self.data_array = input_list
		self.data_list = self._flatten_list(self.data_array)
		# TODO: data in memoryview still missing

	@staticmethod
	def _is_base_dtype(elem: any) -> bool:
		return isinstance(elem, int) or isinstance(elem, float)

	@classmethod
	def _flatten_list(cls, lst: list | int | float) -> list:
		"""
		Flatten list (C or Fortran style not yet registered)
		"""
		if cls._is_base_dtype(lst):
			return [lst]

		flattened_list = []
		for elem in lst:
			flattened_list += cls._flatten_list(elem)

		return flattened_list

	@staticmethod
	def _multiply_int_list(lst: list[int]) -> int:
		accum = 1
		for elem in lst:
			accum *= elem
		return accum

	# TODO: evaluate empty list edge case
	@classmethod
	def _get_list_shape(
		cls,
		test_list: int | float | list,
	) -> tuple[tuple[int], type[int] | type[float] | type[None]]:
		"""
		Check if the list has consistent dimensions and types.
		In numpy jargon, checks the homogeneity of the input list.
		"""

		# if the current object is a list, checks recursively for the
		# shape of all sublists. If these are consistent, then returns
		# its shape, prepended to the shape of the sublists.
		if isinstance(test_list, list):
			length = len(test_list)
			if length == 0:
				return (0, None)

			shape_dtype_list = [cls._get_list_shape(elem) for elem in test_list]
			first_shape = shape_dtype_list[0][0]
			first_dtype = shape_dtype_list[0][1]
			for shape, dtype in shape_dtype_list[1:]:
				if shape != first_shape:
					raise ValueError("Inconsistent shape between sublists")
				if not (first_dtype == dtype):
					raise ValueError("Inconsistent typing between sublists' elements")
			return (length, *first_shape), first_dtype

		if isinstance(test_list, int) or isinstance(test_list, float):
			return (), type(test_list)

		else:
			return ValueError("Cannot get the shape of non-list, non-nummber type")

	# From the memoryview, with a sanity check on the current shape
	# and checking numpy's behaviour, it should be easy
	def reshape(self, new_shape: tuple[int]):
		pass

	# From its current shape, and checking that the permutation of
	# dimensions is correct, again, checking numpy's behaviour, it
	# should be easy
	def transpose(self):
		pass

	# check exactly how the pretty printing works here.
	def __str__(self):
		pass


class MiniNumPy:
	"""
	Namespace for all MiniNumPy constructor methods
	"""

	@staticmethod
	def array(list_or_nested_list: list) -> Array:
		"""Creates array from a given list"""
		# Sanitize array
		return Array(list_or_nested_list)

	@classmethod
	def _singular_value_array(cls, shape: tuple[int], value: int | float) -> Array:
		"""Builds an array of the specified shape, with given value."""
		size = 1
		for dim_length in shape:
			size *= dim_length

		current_value = value
		for dim in shape[::-1]:
			if isinstance(current_value, int) or isinstance(current_value, float):
				current_list_level = [current_value for _ in range(dim)]
			else:  # should be list instance. Copies it along new axis.
				current_list_level = [current_value.copy() for _ in range(dim)]
			current_value = current_list_level

		return cls.array(current_list_level)

	@classmethod
	def zeros(cls, shape: tuple[int]) -> Array:
		"""Returns an array of zeros of the specified shape."""
		return cls._singular_value_array(shape, 0)

	@classmethod
	def ones(cls, shape: tuple[int]) -> Array:
		"""Returns an array of ones of the specified shape."""
		return cls._singular_value_array(shape, 1)

	@classmethod
	def eye(cls, n: int) -> Array:
		"""Returns a square array of shape (n,n) with ones in its diagonal."""
		array = cls.zeros((n, n))
		for i in range(n):
			array.data_array[i][i] = 1

		return array

	@staticmethod
	def _check_range(start: float, stop: float) -> None:
		"""Helper method to check validity of range and raise exception otherwise"""
		if start > stop:
			raise RuntimeError("Invalid range, start > stop")

	@staticmethod
	def _check_postive(number: float, name: str) -> None:
		"""Helper method to check validity of positive number and raise exception otherwise"""
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
