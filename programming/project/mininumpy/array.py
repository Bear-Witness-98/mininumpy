# File with


class Array:
	"""Array to implement lite version of Numpy."""

	data: memoryview
	data_list = list
	shape: tuple[int]
	dtype: type[int] | type[float] | type[None]
	ndim: int
	size: int

	@staticmethod
	def _multiply_int_list(lst: list[int]) -> int:
		accum = 1
		for elem in lst:
			accum *= elem
		return accum

	@staticmethod
	def _is_int_or_float(elem: any) -> bool:
		return isinstance(elem, int) or isinstance(elem, float)

	# TODO: evaluate empty list edge case
	@classmethod
	def _get_list_shape(
		cls,
		test_list: int | float | list[int | float | None],
	) -> tuple[tuple[int], type[int] | type[float] | type[None]]:
		"""
		Get the shape and type of a multi-nested list.
		If the dimensions do not form an 'n-dimensional square' (if they are not homogeneous in
		numpy jargon), or if the types do not match, an appropriate exeption will be raised.

		Value error will be raised if non-expected object is inputted to the function.
		"""

		# If the current object is a list, checks recursively for the
		# shapes and types of all sublists.
		#
		# If these shapes and types are consistent, then the main list's shape is its length
		# preppended to the shape of any of its sublists (which should be the same).
		if isinstance(test_list, list):
			length = len(test_list)
			if length == 0:
				return ((0,), None)

			shape_dtype_list = [cls._get_list_shape(elem) for elem in test_list]
			first_shape = shape_dtype_list[0][0]
			first_dtype = shape_dtype_list[0][1]
			for shape, dtype in shape_dtype_list[1:]:
				if shape != first_shape:
					raise ValueError("Inconsistent shape between sublists")
				if first_dtype != dtype:
					raise ValueError("Inconsistent typing between sublists' elements")
			return (length, *first_shape), first_dtype

		if cls._is_int_or_float(test_list):
			return (), type(test_list)

		raise ValueError("Cannot get the shape of non-list, non-number type")

	@classmethod
	def _flatten_list(cls, lst: list | int | float) -> list:
		"""
		Flatten list (check if this may correspond with C or Fortran style memory storing.)
		"""
		if cls._is_int_or_float(lst):
			return [lst]

		flattened_list = []
		for elem in lst:
			flattened_list += cls._flatten_list(elem)

		return flattened_list

	# TODO: get the memoryview thing correctly if necessary.
	def __init__(self, input_list: list[int | float]):
		"""
		Creates Array object from given list,

		This class only accepts same type (int or float) multi-nested arrays of homegenous
		dimension. If the input list does not match these conditions, and exception will be raised
		at runtime.
		"""

		self.shape, self.dtype = self._get_list_shape(input_list)
		self.ndim = len(self.shape)  # in case of empty list input, ndim = 1 (same as numpy)
		self.size = self._multiply_int_list(self.shape)

		# actually store data
		self.data_list = self._flatten_list(input_list)

	def reshape(self, new_shape: tuple[int]):
		if self._multiply_int_list(new_shape) != self.size:
			raise RuntimeError("size of input shape does not correspond to size of current array")
		self.shape = new_shape

	# From its current shape, and checking that the permutation of
	# dimensions is correct, again, checking numpy's behaviour, it
	# should be easy.
	def transpose(self, permutation: tuple[int] | None = None):
		if permutation is None:
			pass
		if set(permutation) is not set(range(len(permutation))):
			raise RuntimeError("Permutation invalid.")

	@classmethod
	def _unflatten_list(
		cls,
		flattened_list: list[int | float | None],
		shape: tuple[int],
	) -> list:
		"""
		Unflattens given list to specified shape.
		"""
		# sanity assertion
		if cls._multiply_int_list(shape) != len(flattened_list):
			raise ValueError("Given list size does not match size obtained from given shape.")

		# empty list and dtype list cases
		if flattened_list == []:
			return flattened_list
		if (len(shape) == 0) or (len(shape) == 1):
			return flattened_list

		# recursive call
		unflattend_list = []
		sublength = cls._multiply_int_list(shape[1:])
		for idx in range(shape[0]):
			unflattend_list.append(
				cls._unflatten_list(
					flattened_list[idx * sublength : (idx + 1) * sublength], shape[1:]
				)
			)

		return unflattend_list

	# for pretty printing purposes
	def __str__(self):
		return self._unflatten_list(self.data_list, self.shape).__str__()

	__repr__ = __str__


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
		# creates 2-dimensional list with proper shape and values
		lst = [
			[1 if idx_2 == idx else 0 for idx_2, __ in enumerate(range(n))]
			for idx, _ in enumerate(range(n))
		]
		# converts to array and returns
		return cls.array(lst)

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
