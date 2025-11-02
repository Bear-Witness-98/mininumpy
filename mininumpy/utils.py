"""
Utility functions for the MiniNumPy module
"""


def assert_type_is_int_float(dtype: any) -> None:
	if dtype not in {int, float}:
		raise ValueError(f"Expected int or float type, got {dtype}")


def assert_list_int_float(input_list: any) -> None:
	if not isinstance(input_list, (int, float, list)):
		raise ValueError(
			"Unexpected input for Array class. \n"
			"Expected array-like list or single float or int value"
		)


def assert_tuple_of_int(input_tuple: any) -> None:
	if not isinstance(input_tuple, tuple):
		raise ValueError(
			"Unexpected input for builder function. \n"
			f"Expected tuple of ints, found {type(input_tuple)}"
		)
	for elem in input_tuple:
		if not isinstance(elem, int):
			raise ValueError(
				"Unexpected input for builder function. \n"
				f"Expected tuple of ints, found {type(elem)}-type in tuple"
			)


def assert_list_of_float_int(input_lst: any) -> None:
	if not isinstance(input_lst, list):
		raise ValueError(
			f"Unexpected input for builder function. \nExpected list , found {type(input_lst)}"
		)
	for elem in input_lst:
		if not isinstance(elem, (int, float)):
			raise ValueError(
				"Unexpected input for builder function. \n"
				f"Expected tuple of ints or floats, found {type(elem)}-type in list"
			)
