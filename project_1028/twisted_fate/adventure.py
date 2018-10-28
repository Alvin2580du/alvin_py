# -*- coding: utf-8 -*-
"""
Text Adventure Game
An adventure in making adventure games.

To test your current solution, run the `test_my_solution.py` file.

Refer to the instructions on Canvas for more information.

"I have neither given nor received help on this assignment."
author: YOUR NAME HERE
"""
__version__ = 8

# 2) print_introduction: Print a friendly welcome message for your game

# 3) get_initial_state: Create the starting player state dictionary

# 4) print_current_state: Print some text describing the current game world

# 5) get_options: Return a list of commands available to the player right now

# 6) print_options: Print path_to_move the list of commands available    

# 7) get_user_input: Repeatedly prompt the user to choose a valid command

# 8) process_command: Change the player state dictionary based on the command

# 9) print_game_ending: Print a victory, lose, or quit message at the end    

# Command Paths to give to the unit tester
WIN_PATH = "D:\\alvin_py\\stepTwo\\600"
LOSE_PATH = "D:\\alvin_py\\stepTwo\\600"


# 1) Main function that runs your game, read it over and understand
# how the player state flows between functions.
def main():
    # Print an introduction to the game
    print_introduction()
    # Make initial state
    the_player = get_initial_state()
    # Check victory or defeat
    while the_player['game status'] == 'playing':
        print("-- "*20)
        # Give current state
        print_current_state(the_player)
        # Get options
        available_options = get_options(the_player)
        # Give next options
        print_options(available_options)
        # Get Valid User Input
        chosen_command = get_user_input(available_options)
        # Process Commands and change state
        process_command(chosen_command, the_player)
        print(the_player)
    # Give user message
    print_game_ending(the_player)


# Executes the main function
def print_introduction():
    print("Welcome to our came!")


def get_initial_state():
    return {'game status': 'playing', 'location': "Yard", 'has key': False, 'Score': 0}


def print_current_state(the_player):
    print(the_player['location'])


def get_options(the_player):
    """
     returns a list of strings representing commands available to the user right now 
     (e.g., places to move to, commands to execute, things available to be collected).
     you could check the player's location and offer them other locations to move to. 
    :param the_player: 
    :return: 
    """
    path_to_move = []
    if the_player['location'] == 'Yard':
        res = ['Enter house', 'Leave yard', 'quit']
        path_to_move += res

    if the_player['location'] == 'Forest':
        res = ['Enter yard', 'Get key', 'quit']
        path_to_move += res

    if the_player['location'] == 'Living Room':
        res = ['Leave house', 'Go upstairs', 'Dining Room', 'quit']
        path_to_move += res

    if the_player['location'] == 'Dining Room':
        res = ['Eat Food', 'Living Room', 'quit']
        path_to_move += res

    if the_player['location'] == 'Upstairs':
        res = ['Bedroom', 'Go downstairs', 'quit']
        path_to_move += res

    return path_to_move


def print_options(options):
    print("print_options:{}".format(",".join(options)))


def get_user_input(available_options):
    x = input("Please input which command from {} you want to choose: ".format(",".join(available_options)))
    return x


def process_command(chosen_command, the_player):
    print("120 the_player:{}".format(the_player))

    if chosen_command == 'quit':
        the_player['game status'] = 'quit'

    else:
        if chosen_command == 'Leave yard':
            the_player['location'] = 'Forest'
            the_player['game status'] = 'playing'
            the_player['has key'] = False
        if chosen_command == 'Enter yard':
            the_player['location'] = 'Yard'
            the_player['game status'] = 'playing'
            if the_player['has key']:
                the_player['has key'] = True
            else:
                the_player['has key'] = False
        if chosen_command == 'Get key':
            the_player['location'] = 'Forest'
            the_player['game status'] = 'playing'
            the_player['has key'] = True

        if chosen_command == 'Enter house':
            the_player['location'] = 'Living Room'
            the_player['game status'] = 'playing'
            if the_player['has key']:
                the_player['has key'] = True
            else:
                the_player['has key'] = False

        if chosen_command == 'Leave house':
            the_player['location'] = 'Yard'
            the_player['game status'] = 'playing'
            if the_player['has key']:
                the_player['has key'] = True
            else:
                the_player['has key'] = False

        if chosen_command == 'Dining Room':
            the_player['location'] = 'Dining Room'
            the_player['game status'] = 'playing'
            if the_player['has key']:
                the_player['has key'] = True
            else:
                the_player['has key'] = False
        if chosen_command == 'Living Room':
            the_player['location'] = 'Living Room'
            the_player['game status'] = 'playing'
            if the_player['has key']:
                the_player['has key'] = True
            else:
                the_player['has key'] = False

        if chosen_command == 'Eat Food':
            the_player['location'] = "Lose"
            the_player['game status'] = 'lost'
            the_player['has key'] = False
        if chosen_command == 'Go upstairs':
            the_player['location'] = 'Upstairs'
            the_player['game status'] = 'playing'
            if the_player['has key']:
                the_player['has key'] = True
            else:
                the_player['has key'] = False
        if chosen_command == 'Go downstairs':
            the_player['location'] = 'Living Room'
            the_player['game status'] = 'playing'
            if the_player['has key']:
                the_player['has key'] = True
            else:
                the_player['has key'] = False
        if chosen_command == 'Bedroom':
            if the_player['has key']:
                the_player['game status'] = 'won'
            else:
                the_player['location'] = "Upstairs"
                the_player['game status'] = 'playing'
                if the_player['has key']:
                    the_player['has key'] = True
                else:
                    the_player['has key'] = False


def print_game_ending(the_player):
    if the_player['game status'] == 'quit':
        print("quit the game")
    elif the_player['game status'] == 'won':
        print("Congratulations, Victory!")
    elif the_player['game status'] == 'lost':
        print("sorry, you are lost!")
    else:
        print("Not imm")

if __name__ == "__main__":
    '''
    You might comment path_to_move the main function and call each function
    one at a time below to try them path_to_move yourself '''
    # main()
    ## e.g., comment path_to_move main() and uncomment the line(s) below
    # print_introduction()
    # print(get_initial_state())
    # ...
    command_tests = [([], ['home', 'second door', 'upstairs'], 'home'),
                     ([], ['open the door'], 'open the door'),
                     (['hide', 'run'], ['attack', 'defend'], 'attack'),
                     (['wait', 'act', 'item'], ['mercy'], 'mercy'),
                     (['quit'], ['home'], 'quit'),
                     (['wait', 'act', 'item', 'quit'], ['home'], 'quit'),
                     (['wait', 'quit'], ['mercy'], 'quit')]
    for bad_commands, command_test, expected in command_tests:
        print(command_test)