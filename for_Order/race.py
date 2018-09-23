#####################################
# race.py
# Homework 2 Problem 2
#####################################

'''
A simple turtle race game
'''

import turtle
'''import turtle to use turtle drawing commands'''

import random as rd
'''import random to get access to the random module'''

'''
Initialize the window
DO NOT MODIFY
'''
def initWindow():
    window = turtle.Screen()
    window.setup(width = 610, height = 610)
    window.bgcolor("black")
    window.setworldcoordinates(0, 0, 609, 609)

    return window

'''
Draw a thick horizontal line
DO NOT MODIFY
'''
def drawLine(aTurtle, color, x, y, length):
    aTurtle.pencolor(color)
    aTurtle.up()
    aTurtle.setposition(x, y)
    aTurtle.down()
    aTurtle.forward(length)

'''
Initialize and return a single turtle
based on description in homework
'''
def initTurtle(x, y, color):
    t = turtle.Turtle()
    t.color(color)
    t.shape("turtle")
    t.penup()
    t.goto(x, y)
    t.left(90)
    return t
    

'''
Move a turtle based on description
in homeowrk
'''
def moveTurtle(aTurtle):
    aTurtle.forward(rd.randint(-5, 12))
    aTurtle.left(rd.randint(-4, 4))

    

def main():
    window = initWindow()
    
    # Create a turtle to draw start and finish lines
    drawTurtle = turtle.Turtle()
    drawTurtle.hideturtle()
    drawTurtle.pensize(5)
    drawTurtle.speed(0)

    # Actually draw start and finish lines
    drawLine(drawTurtle, "dark goldenrod", 0, 100, 600)
    drawLine(drawTurtle, "sky blue", 0, 430, 600)

    # setup four turtles with different colors
    # and put them equally spaced apart below the goldish-brown line
    firstTurtle = initTurtle(120, 80, 'red')
    secondTurtle = initTurtle(240, 80, 'yellow')
    thirdTurtle = initTurtle(360, 80, 'pink')
    fourthTurtle = initTurtle(480, 80, 'white')

    for i in range (100):
        moveTurtle(firstTurtle)
        moveTurtle(secondTurtle)
        moveTurtle(thirdTurtle)
        moveTurtle(fourthTurtle)

    window.exitonclick()

main()
    




    
