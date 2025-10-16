# File with the implementation of the array type.
from __future__ import annotations  # for typehinting the Array class within itself

from math import exp, log, sqrt


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
		"""
		An empty list here outputs 1
		"""
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
		Flatten list.
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
		dimension. If the input list does not match these conditions, a ValueError will be raised
		at runtime.
		"""

		self.shape, self.dtype = self._get_list_shape(input_list)
		self.ndim = len(self.shape)  # in case of empty list input, ndim = 1 (same as numpy)
		self.size = self._multiply_int_list(self.shape)

		# actually store data
		self.data_list = self._flatten_list(input_list)

	def copy(self) -> Array:
		"""
		Make a copy of the current instance.
		"""
		new_array = Array([])
		# populate it with current values
		new_array.data_list = self.data_list.copy()
		new_array.shape = tuple(elem for elem in self.shape)
		new_array.dtype = self.dtype
		new_array.ndim = self.ndim
		new_array.size = self.size

		return new_array

	def reshape(self, new_shape: tuple[int]) -> Array:
		"""
		Reshapes array to given new_shape.

		If the size of new_shape does not match self.size, a RuntimeError will be rised.
		"""
		if self._multiply_int_list(new_shape) != self.size:
			raise RuntimeError("size of input shape does not correspond to size of current array")
		new_array = self.copy()
		new_array.shape = new_shape

		return new_array

	@classmethod
	def _flattened_idx(cls, idx: tuple[int], shape: tuple[int]) -> int:
		"""
		Converts an multidimensional index into a linear one for internal data representation.

		If (d_0, ..., d_(k-1)) is the shape of the array, then a multidimensional index is a tuple
		I = (i_0, ..., i_(k-1)) with i_j in the range (0, ..., d_j).

		The corresponding linear index for I is:
		lin_idx_(k+1) = i_k + d_k( i_(k-1) + d_(k-1)( ... i_1 + d_1(i_0) ...))

		This linearized index for a k-dimensional array can be computed recursively by the
		recursive sequence:
		lin_idx_0 = i_0
		lin_idx_n = i_n + d_n*lin_idx_(n_1)

		Notice that for these computations
		"""
		# TODO: should do a sanity check here
		if len(idx) == 1:
			return idx[-1]
		return idx[-1] + (shape[-1] * cls._flattened_idx(idx[:-1], shape[:-1]))

	@classmethod
	def _circular_increment_idx(cls, idx: tuple[int], shape: tuple[int]) -> tuple[int]:
		"""
		Increment the multidimensional idx of an array in 1.

		This operation is defined so that it adds one to the last idx position, and in case
		of overflow, it carries a +1 to the previous idx position.

		It is implemented in a circular manner for implementation reasons.
		"""
		if len(idx) == 0:
			return ()

		if idx[-1] < (shape[-1] - 1):
			return *idx[:-1], (idx[-1] + 1)

		return *cls._circular_increment_idx(idx[:-1], shape[:-1]), 0

	def transpose(self, permutation: tuple[int] | None = None) -> Array:
		# sanity check the input
		if permutation is None:
			permutation = tuple(reversed(range(self.ndim)))
		if not isinstance(permutation, tuple):
			raise ValueError("Non tuple, non-None value given as permutation")
		if set(permutation) != set(range(len(permutation))):
			raise RuntimeError("Invalid permutation for transposition.")

		# get new shape and empty new data. Same behaviour as numpy.
		new_shape = tuple(self.shape[p] for p in permutation)
		new_data = [0 for _ in range(len(self.data_list))]

		idx = tuple(0 for _ in range(self.ndim))
		for _ in range(self.size):
			# get index in list for value
			linear_idx = self._flattened_idx(idx, self.shape)

			# compute corresponding idx in new array, and its linearization
			new_idx = tuple(idx[p] for p in permutation)
			new_linear_idx = self._flattened_idx(new_idx, new_shape)

			# copy corresponding values and increment idx
			new_data[new_linear_idx] = self.data_list[linear_idx]
			idx = self._circular_increment_idx(idx, self.shape)

		# Create new_array and return it with new shape and list
		new_array = self.copy()
		new_array.shape = new_shape
		new_array.data_list = new_data

		return new_array

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
	def __str__(self) -> str:
		return self._unflatten_list(self.data_list, self.shape).__str__()

	__repr__ = __str__

	# elementwise operations
	def exp(self) -> Array:
		"""
		Return a copy of the array with elements e^(elem).
		"""
		new_array = self.copy()
		new_array.data_list = [exp(elem) for elem in new_array.data_list]
		new_array.dtype = float
		return new_array

	def log(self) -> Array:
		"""
		Return a copy of the array with elements log_e(elem).
		"""
		new_array = self.copy()
		new_array.data_list = [log(elem) for elem in new_array.data_list]
		new_array.dtype = float
		return new_array

	def sqrt(self) -> Array:
		"""
		Return a copy of the array with elements sqrt(elem).
		"""
		new_array = self.copy()
		new_array.data_list = [sqrt(elem) for elem in new_array.data_list]
		new_array.dtype = float
		return new_array

	def abs(self) -> Array:
		"""
		Return a copy of the array with elements abs(elem).
		"""
		new_array = self.copy()
		new_array.data_list = [abs(elem) for elem in new_array.data_list]
		return new_array

	# binary operations
	@staticmethod
	def _broadcast_shapes(shape1: tuple[int], shape2: tuple[int]) -> bool:
		"""
		Returns the shape into which both input shapes could be broadcasted.
		Raises ValueError if the shapes cannot be broadcasted together.

		The condition is that, starting from the rightmost dimension either:
		1) the dimensions must match
		2) one of the dimensions is 1

		One can extend a shape with trailing ones up to the length of the
		shape with more dimensions.
		"""
		broadcasted_shape = ()
		smaller_array, bigger_array = (
			(shape1, shape2) if (len(shape1) < len(shape2)) else (shape2, shape1)
		)
		min_length = len(smaller_array)
		max_length = len(bigger_array)

		# check if the corresponding rightmost dimensions match, and store them.
		for idx in range(min_length):
			dim = -1 - idx
			# this could also be ({shape1[dim]} xor {shape2[dim]} xor {1}) != {}
			is_dim_compatible = (shape1[dim] == shape2[dim]) or {1}.intersection(
				{shape1[dim], shape2[dim]}
			)
			if not is_dim_compatible:
				raise ValueError(
					f"Arrays of shape {shape1} and {shape2} cannot be broadcasted together."
				)
			broadcasted_shape = (max(shape1[dim], shape2[dim]),) + broadcasted_shape

		# copy the remaining dims of the larger shape
		remainin_dims = bigger_array[0 : max_length - min_length]
		broadcasted_shape = remainin_dims + broadcasted_shape
		return broadcasted_shape

	def _broadcast_back_multi_idx_to_shape(
		self,
		multi_idx: tuple[int],
		shape: tuple[int],
	) -> tuple[int]:
		starting_dim = len(multi_idx) - len(shape)
		new_mulit_idx = list(multi_idx[starting_dim:])
		for idx, (dim, dim_value) in enumerate(zip(shape, new_mulit_idx)):
			if dim_value > dim:
				new_mulit_idx[idx] = 1
		return tuple(new_mulit_idx)

	def _operation_with_broadcasting(
		self,
		array1: Array,
		array2: Array,
		new_dtype: type[int] | type[float] | type[None],
	) -> None:
		# get shape of, and create new array
		new_array = Array([])
		new_array.shape = self._broadcast_shapes(array1.shape, array2.shape)
		new_array.ndim = len(new_array.shape)
		new_array.size = self._multiply_int_list(new_array.shape)
		new_array.dtype = new_dtype
		new_array.data_list = [0 for _ in range(new_array.size)]

		# create starting multi-idx, and iterate over all possible values
		multi_idx = tuple([0 for _ in new_array.shape])
		print(new_array.shape)
		print("====================================")
		for _ in range(new_array.size):
			# get value for first array
			print("++++++++++++++++++++++++++++++++++++")
			print(multi_idx)
			array1_multi_idx = self._broadcast_back_multi_idx_to_shape(multi_idx, array1.shape)
			array1_idx = self._flattened_idx(array1_multi_idx, array1.shape)
			print(array1_multi_idx)
			print(array1_idx)
			array1_value = array1.data_list[array1_idx]
			# get value for second array
			array2_multi_idx = self._broadcast_back_multi_idx_to_shape(multi_idx, array2.shape)
			array2_idx = self._flattened_idx(array2_multi_idx, array2.shape)
			array2_value = array2.data_list[array2_idx]
			# update value on new array
			new_array_idx = self._flattened_idx(multi_idx, new_array.shape)
			new_array.data_list[new_array_idx] = array1_value + array2_value
			# increase the general idx
			multi_idx = self._circular_increment_idx(multi_idx, new_array.shape)

		return new_array

	def __add__(self, array2: Array) -> Array:
		# return self._operation_with_broadcasting(self, array2, float)
		if self.shape != array2.shape:
			# attempt a broadcast here
			raise ValueError("Arrays of different shapes could not be broadcasted together")
		new_array = self.copy()
		new_array.data_list = [
			elem1 + elem2 for elem1, elem2 in zip(self.data_list, array2.data_list)
		]
		return new_array

	def __sub__(self, array2: Array) -> Array:
		if self.shape != array2.shape:
			# attempt a broadcast here
			raise ValueError("Arrays of different shapes could not be broadcasted together")
		new_array = self.copy()
		new_array.data_list = [
			elem1 - elem2 for elem1, elem2 in zip(self.data_list, array2.data_list)
		]
		return new_array

	def __mul__(self, array2: Array) -> Array:
		if self.shape != array2.shape:
			# attempt a broadcast here
			raise ValueError("Arrays of different shapes could not be broadcasted together")
		new_array = self.copy()
		new_array.data_list = [
			elem1 * elem2 for elem1, elem2 in zip(self.data_list, array2.data_list)
		]
		return new_array

	def __truediv__(self, array2: Array) -> Array:
		if self.shape != array2.shape:
			# attempt a broadcast here
			raise ValueError("Arrays of different shapes could not be broadcasted together")
		new_array = self.copy()
		new_array.data_list = [
			elem1 / elem2 for elem1, elem2 in zip(self.data_list, array2.data_list)
		]
		return new_array

	def __pow__(self, array2: Array) -> Array:
		if self.shape != array2.shape:
			# attempt a broadcast here
			raise ValueError("Arrays of different shapes could not be broadcasted together")
		new_array = self.copy()
		new_array.data_list = [
			elem1**elem2 for elem1, elem2 in zip(self.data_list, array2.data_list)
		]
		return new_array
