from mininumpy.array import MiniNumPy as mnp


def main() -> None:
	print("This is a test")
	array = mnp.array([1, 2, 3, 4])
	print(array.stored_data)

	array_new = mnp.zeros((2, 3))
	print(array_new)


if __name__ == "__main__":
	main()
