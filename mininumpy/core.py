from math import prod

from .array import Array
from .utils import assert_list_int_float, assert_tuple_of_int

"""
Main methods for creating MiniNumPy arrays.
"""


def _get_shape_and_type(
	test_list: int | float | list[int | float | None],
) -> tuple[tuple[int], type[int | float | None]]:
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

		shape_dtype_list = [_get_shape_and_type(elem) for elem in test_list]
		first_shape = shape_dtype_list[0][0]
		first_dtype = shape_dtype_list[0][1]
		for shape, dtype in shape_dtype_list[1:]:
			if shape != first_shape:
				raise ValueError("Inconsistent shape between sublists")
			if dtype != first_dtype:
				raise ValueError("Inconsistent typing between sublists' elements")
		return (length, *first_shape), first_dtype

	# if it is just a number, return empty tuple and type
	if isinstance(test_list, (int, float)):
		return (), type(test_list)

	raise ValueError("Cannot get the shape of non-list, non-number type")


def _flatten_list(lst: list | int | float) -> list:
	"""
	Flattens multi-nested list of int or floats.
	"""
	if isinstance(lst, (int, float)):
		return [lst]

	flattened_list = []
	for elem in lst:
		flattened_list += _flatten_list(elem)

	return flattened_list


def array(list_or_nested_list: list) -> Array:
	"""Creates array from a given list"""
	# Sanitize input
	assert_list_int_float(list_or_nested_list)

	lst_shape, lst_dtype = _get_shape_and_type(list_or_nested_list)
	lst_flattend = _flatten_list(list_or_nested_list)

	return Array(lst_shape, lst_dtype, data=lst_flattend)


def _singular_value_array(shape: tuple[int], value: int | float) -> Array:
	"""Builds an array of the specified shape, with given value."""

	# computes needed values to init a new array
	dtype = type(value)

	size = prod(shape)
	data = [value for _ in range(size)]

	return Array(shape, dtype, data)


def zeros(shape: tuple[int]) -> Array:
	"""Returns an array of zeros of the specified shape."""
	assert_tuple_of_int(shape)
	return _singular_value_array(shape, 0)


def ones(shape: tuple[int]) -> Array:
	"""Returns an array of ones of the specified shape."""
	assert_tuple_of_int(shape)
	return _singular_value_array(shape, 1)


def eye(n: int) -> Array:
	"""Returns a square array of shape (n,n) with ones in its diagonal."""
	if not isinstance(n, int):
		raise ValueError(f"Expected int, but {type(n)} was given")

	# creates linear array of 1 values in the positions to be moved to the diagonal
	lst = [1 if outer - inner % n == 0 else 0 for outer in range(n) for inner in range(n)]
	return Array((n, n), int, lst)


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
