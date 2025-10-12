# File with the implementation of the array type.


class Array:
	"""Array to implement lite version of Numpy."""

	data: memoryview
	data_list: list
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
				if dtype != first_dtype:
					raise ValueError("Inconsistent typing between sublists' elements")
			return (length, *first_shape), first_dtype

		# if it is just a number, return empty tuple and type
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

		return self

	@classmethod
	def _flattened_idx(cls, idx: tuple[int], shape: tuple[int]) -> int:
		# TODO: should do a sanity check here
		if len(idx) == 1:
			return idx[-1]
		return idx[-1] + (shape[-1] * cls._flattened_idx(idx[:-1], shape[:-1]))

	@classmethod
	def _circular_increment_idx(cls, idx: tuple[int], shape: tuple[int]) -> tuple[int]:
		if len(idx) == 0:
			return ()

		if idx[-1] < (shape[-1] - 1):
			return *idx[:-1], (idx[-1] + 1)

		return *cls._circular_increment_idx(idx[:-1], shape[:-1]), 0

	# From its current shape, and checking that the permutation of
	# dimensions is correct, again, checking numpy's behaviour, it
	# should be easy.
	def transpose(self, permutation: tuple[int] | None = None):
		if permutation is None:
			permutation = tuple(reversed(range(self.ndim)))

		if set(permutation) != set(range(len(permutation))):
			raise RuntimeError("Invalid permutation for transposition.")

		# permutation has same behaviour as numpy.
		new_shape = tuple(self.shape[p] for p in permutation)

		# initialize empty list for the new, reordered values.
		new_data = [0 for _ in range(len(self.data_list))]

		idx = tuple(0 for _ in range(self.ndim))
		for _ in range(self.size):
			linear_idx = self._flattened_idx(idx, self.shape)
			idx_permutation = tuple(idx[p] for p in permutation)
			permuted_linear_idx = self._flattened_idx(idx_permutation, new_shape)
			new_data[permuted_linear_idx] = self.data_list[linear_idx]
			idx = self._circular_increment_idx(idx, self.shape)

		# update with newly computed variables
		self.shape = new_shape
		self.data_list = new_data

		return self

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
