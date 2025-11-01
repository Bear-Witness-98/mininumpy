"""
Utility functions for the MiniNumPy module
"""


def assert_list_int_float(input_list: any) -> None:
	if not isinstance(input_list, (int, float, list)):
		raise ValueError(
			"Unexpected input for Array class. \n"
			"Expected array-like list or single float or int value"
		)


def assert_int_tuple(input_tuple: any) -> None:
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
