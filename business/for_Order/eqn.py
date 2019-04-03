inputs = input('Please input equation:')


def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def get_operator(func):
    if "+" in func:
        return "add"

    if '-' in func:
        return 'sub'


def get_index_of_x(func):
    if 'x' in func:
        return func.index('x')
    if "X" in func:
        return func.index("X")


def main():
    func = inputs.replace(" ", "")

    operator = get_operator(func)
    index = get_index_of_x(func)
    # add
    if operator == 'add' and index == 0:
        x = subtract(int(func[-1]), int(func[-3]))
        print("the value of X:{}".format(x))

    if operator == 'add' and index == 2:
        x = subtract(int(func[-1]), int(func[0]))
        print("the value of X:{}".format(x))

    if operator == 'add' and index == 4:
        x = add(int(func[0]), int(func[2]))
        print("the value of X:{}".format(x))

    # subtract
    if operator == 'sub' and index == 0:
        x = add(int(func[-1]), int(func[2]))
        print("the value of X:{}".format(x))

    if operator == 'sub' and index == 2:
        x = subtract(int(func[0]), int(func[-1]))
        if x < 0:
            print("Please input non-negative integers")
        else:
            print("the value of X:{}".format(x))

    if operator == 'sub' and index == 4:
        x = subtract(int(func[0]), int(func[2]))
        if x < 0:
            print("Please input non-negative integers")
        else:
            print("the value of X:{}".format(x))

main()
