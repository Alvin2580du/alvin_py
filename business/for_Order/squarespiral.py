import turtle


def main():
    t = turtle.Pen()
    turtle.bgcolor("black")
    sides = 4
    t.seth(180)
    for x in range(1, 361):
        t.pencolor("red")
        t.forward(x)
        t.left(360 / sides - 1)

main()
