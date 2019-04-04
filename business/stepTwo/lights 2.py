"""Question 2 Part e
List is collection of numbers while String is sets of characters.
List is able to be motified after we created it.
String is only able to change by operators.
If we want to put a String to a List, we need to conver this String to a number by using the split() method. 
"""
import random


def genBoard(bsize):
    lightList = []
    for i in range(bsize):
        lightList.append(random.randint(0, 1))
    return lightList


def toggle(board, pos):
    length = len(board)
    if pos == 0:
        if board[0] == 1:
            board[0] = 0
            if board[1] == 1:
                board[1] = 0
            else:
                board[1] = 1
        else:
            board[0] = 1
            if board[1] == 1:
                board[1] = 0
            else:
                board[1] = 1

    if 0 < pos <= length-2:
        if board[pos] == 1:
            board[pos] = 0
            if board[pos-1] == 1:
                board[pos-1] = 0
            else:
                board[pos-1] = 1
            if board[pos+1] == 1:
                board[pos+1] = 0
            else:
                board[pos+1] = 1
        else:
            board[pos] = 1
            if board[pos-1] == 1:
                board[pos-1] = 0
            else:
                board[pos-1] = 1
            if board[pos+1] == 1:
                board[pos+1] = 0
            else:
                board[pos+1] = 1
    else:
        if board[length-1] == 1:
            board[length-1] = 0
            if board[length-2] == 1:
                board[length-2] = 0
            else:
                board[length-2] = 1
        else:
            board[length-1] = 1
            if board[length-2] == 1:
                board[length-2] = 0
            else:
                board[length-2] = 1
    return board


def checkWin(aboard):
    if 1 in aboard:
        return False
    else:
        return True


def main():
    bsize = 6
    board = genBoard(bsize)
    print(board)
    for i in range(30):
        p = int(input("Which position you want to toggle: "))
        if p > bsize-1:
            print("Error")
        board2 = toggle(board, pos=p)
        print(board2)
        print(checkWin(board2))
        if checkWin(board2):
            print("Congratulations!!! You Won!!!")
            break

main()
