# -*- coding: utf-8 -*-
"""
Goblins Magical Loan System
A sophisticated system for approving loans to wizards starting businesses.

To test your current solution, run the `test_my_solution.py` file.

Refer to the instructions on Canavs for more information.

"I have neither given nor received help on this assignment."
author: YOUR NAME HERE
"""
__version__ = 4


# (1) main: Calls the other functions and controls data flow
def main():
    '''
    Main function that calls all the other functions and provides
    an interactive loan processing experience.
        
    Args:
        None
    Returns:
        None
    '''
    print_introduction()
    name = input_name()
    rating = calculate_rating(name)
    print_rating(rating)
    loan_amount = input_loan_amount()
    print_loan_availability(rating=rating, loan_amount=loan_amount)
    print_conclusion()


# (2) print_introduction: Print a welcome message
def print_introduction():
    print("* 1) Introduction ********************************")
    print("Welcome to the Goblins Magical Loan System!")
    print("Please answer the following questions truthfully,")
    print("and we will process your loan request.")
    print("Imposters will be fed to the dragons.")


# (3) input_name: Get the user's name
"""
Please enter your full, legal name.
Magical verification will verify your identity.
Write your name and press enter: 
Welcome, George and Fred Weasley!
George and Fred Weasley,1000,False
"""


def input_name():
    print("* 2) Name ****************************************")
    # print("Please enter your full, legal name.")
    # print("Magical verification will verify your identity.")
    name = input("Please enter your full, legal name.\nMagical verification will verify your identity.\nWrite your name and press enter: ")
    res = name.split(",")[0]
    print("Welcome, {}!".format(res))
    return name


# (4) calculate_rating: Get the user's rating (by Grindlehook)
# The following is Grindlehook's function. Do not modify it.
# You should not worry about HOW it works, but instead think of its
# arguments and return value. Remember you can only call it once!
# Do NOT change the following function definition
def calculate_rating(name):
    '''
    Returns the customer's credit rating, according to the bank's current
    status, the customer, and the alignment of the stars. This function
    is a little delicate, and will break after being called once.

    NOTE (ghook@1/15/2018): DO NOT TOUCH THIS, I FINALLY GOT IT WORKING.

    Returns:
        int: An integer (0-9) representing the customer's credit rating.
    '''
    c = calculate_rating;
    setattr(c, 'r', lambda: setattr(c, 'o', True))
    j = {};
    y = j['CELESTIAL_NAVIGATION_CONSTANT'] = 10
    j[True] = 'CELESTIAL_NAVIGATION_CONSTANT'
    x = str(''[:].swapcase);
    y = y + 11, y + 9, y + -2, y + -2, y + 4, y + -5, y + -1, y + 11, y + 9, \
        y + -6, y + -6, y + -1, y + -5, y + 3, y + -7, y + 7, y + -1, y + -5, y + 8, y + -7, y + 11, y + 1
    z = lambda x, t, o=0: ''.join(map(lambda j: x.__getitem__(j + o), t))
    if hasattr(c, 'o') and not getattr(c, 'o'): return z(x, y)
    c.o = False;
    j['CELESTIAL_NAVIGATION_CONSTANT'].bit_length
    d = (lambda: (lambda: None))()();
    g = globals()
    while d: g['X567S-lumos-17-KLAUS'] = ((d) if (lambda: None)else(j))
    p = lambda p: sum(map(int, list(str(p))))
    MGC = p(sum(map(lambda v: v[0] * 8 + ord(v[1]), enumerate(name))))
    while MGC > 10: MGC = p(MGC)
    if c: return MGC


# Do NOT change the preceding function definition


# (5) print_rating: Print the user's rating
def print_rating(rating):
    print("* 3) Rating **************************************")
    print("Your user rating has been calculated.")
    print("Your rating is: {}/10 points".format(rating))


# (6) input_loan_amount: Get the user's desired loan amount
def input_loan_amount():
    print("* 4) Loan ****************************************")
    print("Loans are made based on your customer rating.")
    print("However, you can request a loan of any size.")
    print("How many galleons do you want?")
    loan_amount = input("Write your loan amount: ")
    return int(loan_amount)


# (7) print_loan_availability: Print whether the loan is available
def print_loan_availability(rating, loan_amount):
    print("* 5) Available? **********************************")
    if test_loan(rating, loan_amount):
        print("Your loan request has been evaluated.")
        print("Loan available: True")
    else:
        print("Your loan request has been evaluated.")
        print("Loan available: False")


# (8) test_loan: Determines if the loan is available
def test_loan(rating, loan_amount):
    if rating * rating * 100 >= int(loan_amount):
        return True
    else:
        return False


# (9) print_conclusion: Prints a thank-you message.
def print_conclusion():
    print('''* 6) Conclusion **********************************
Thanks for using Goblins Magical Loan System!
Best of luck with your new small business.
We hope you'll use Goblins for all your future
banking needs. Remember: Fortius Quo Fidelius!
**************************************************''')


# Call main function only if this file is being run directly
#   as opposed to being run by `test_my_solution.py`
if __name__ == '__main__':
    print("################################################################################")
    main()
