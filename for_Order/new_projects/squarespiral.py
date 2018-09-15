import turtle


def main():
    t = turtle.Pen()
    turtle.bgcolor("black")
    sides = 4
    for x in range(1, 361):
        t.pencolor("red")
        t.forward(x * 3 / sides + x)
        t.left(360 / sides - 2)


main()
