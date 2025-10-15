from .array import Array

"""
Would be nice to put all this under a unique class. Check later how to do so.
"""
# class MiniNumPy:
# 	"""
# 	Namespace for all MiniNumPy constructor methods
# 	"""


def array(list_or_nested_list: list) -> Array:
	"""Creates array from a given list"""
	# Sanitize array
	return Array(list_or_nested_list)


def _singular_value_array(shape: tuple[int], value: int | float) -> Array:
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

	return array(current_list_level)


def zeros(shape: tuple[int]) -> Array:
	"""Returns an array of zeros of the specified shape."""
	return _singular_value_array(shape, 0)


def ones(shape: tuple[int]) -> Array:
	"""Returns an array of ones of the specified shape."""
	return _singular_value_array(shape, 1)


def eye(n: int) -> Array:
	"""Returns a square array of shape (n,n) with ones in its diagonal."""
	# creates 2-dimensional list with proper shape and values
	lst = [
		[1 if idx_2 == idx else 0 for idx_2, __ in enumerate(range(n))]
		for idx, _ in enumerate(range(n))
	]
	# converts to array and returns
	return array(lst)


def _check_range(start: float, stop: float) -> None:
	"""Helper method to check validity of range and raise exception otherwise"""
	if start > stop:
		raise RuntimeError("Invalid range, start > stop")


def _check_positive(number: float, name: str) -> None:
	"""Helper method to check validity of positive number and raise exception otherwise"""
	if number <= 0:
		raise RuntimeError(f"Invalid value ( <=0 ) for {name}")


def arange(start: float, stop: float, step: float) -> Array:
	# Array of values from [start,stop), with difference of
	# step in between each pair

	# sanitize input
	_check_range(start, stop)
	_check_positive(step, "step")

	base_list = []
	idx = 0
	while start + idx * step < stop:
		base_list.append(start + step * idx)
		idx += 1

	return array(base_list)


def linspace(start: float, stop: float, num: int) -> Array:
	# evenly num-spaced values in the interval [start,stop)

	# sanitize input
	_check_range(start, stop)
	_check_positive(num, "num")

	diff = (stop - start) / num
	base_list = []
	for idx in range(num):
		base_list.append(start + diff * idx)

	return array(base_list)


# element-wise operations
def exp(array: Array) -> Array:
	"""
	Return a copy of the array with elements e^(elem).
	"""
	return array.exp()


def log(array: Array) -> Array:
	"""
	Return a copy of the array with elements log_e(elem).
	"""
	return array.log()


def sqrt(array: Array) -> Array:
	"""
	Return a copy of the array with elements sqrt(elem).
	"""
	return array.sqrt()


def abs(array: Array) -> Array:
	"""
	Return a copy of the array with elements abs(elem).
	"""
	return array.abs()
