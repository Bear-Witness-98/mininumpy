# test my functions against numpy
import numpy as np
from numpy import ndarray

import mininumpy as mnp
from mininumpy.array import Array

lst_3d = [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
lst_2d = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
lst_empty = [[], []]
single_number = 7


def _check_equality(mnp_array: Array, np_array: ndarray) -> None:
	pass


def test_3d_list():
	lst_3d_mnp = mnp.array(lst_3d)
	lst_3d_np = np.array(lst_3d)
	_check_equality(lst_3d_mnp, lst_3d_np)


def test_2d_list():
	lst_2d_mnp = mnp.array(lst_2d)
	lst_2d_np = np.array(lst_2d)
	_check_equality(lst_2d_mnp, lst_2d_np)


def test_empy_list():
	lst_empty_mnp = mnp.array(lst_empty)
	lst_empty_np = np.array(lst_empty)
	_check_equality(lst_empty_mnp, lst_empty_np)


def test_normal_list():
	single_number_mnp = mnp.array(single_number)
	single_number_np = np.array(single_number)
	_check_equality(single_number_mnp, single_number_np)
