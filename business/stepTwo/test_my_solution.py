'''
This file is used to test your implementation of `banking.py` for
project 2 (Magical Banking).

You should not modify this file, because the instructor will be running
your code against the original version anyway.

@author: acbart
'''
__version__ = 4
import unittest
from unittest.mock import patch
import sys
import os
import io
import re
import hashlib

TOLERANCE = 2
SAMPLE_OUTPUT_HASH = "0d4fa634b9855eca9dd342854d641edb"

# Ensure the user has created the main.py file
try:
    import banking as student_main
except ImportError:
    raise Exception(('Error! Could not find a "banking.py" file. '
                     'Make sure that there is a "banking.py" in the same '
                     'directory as "{0}"! Spelling is very '
                     'important here.').format(__file__))

# Make sure we are using the right Python version.
PYTHON_3 = sys.version_info >= (3, 0)
if not PYTHON_3:
    raise Exception("This code is expected to be run in Python 3.x")


def make_inputs(*input_list):
    '''
    Helper function for creating mock user input.
    
    Params:
        input_list (list of str): The list of inputs to be returned
    Returns:
        function (str=>str): The mock input function that is returned, which
                             will return the next element of input_list each
                             time it is called.
    '''
    generator = iter(input_list)

    def mock_input(prompt=''):
        print(prompt)
        return next(generator)

    return mock_input


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


def skip_unless_callable(function_name):
    '''
    Helper function to test if the student has defined the relevant function.
    This is meant to be used as a decorator.
    
    Params:
        function_name (str): The name of the function to test in student_main.
    Returns:
        function (x=>x): Either the identity function (unchanged), or a
                         unittest.skip function to let you skip over the
                         function's tests.
    '''
    if hasattr(student_main, function_name):
        if callable(getattr(student_main, function_name)):
            return lambda func: func
        print("You have not defined `{}` in banking.py as a function.".format(function_name))
        return unittest.skip("You have not defined `{}` in banking.py as a function.".format(function_name))
    print("You have not defined `{}` in banking.py as a function.".format(function_name))
    return unittest.skip("You have not defined the `{}` function in banking.py.".format(function_name))


# Verify output file's integrity and accessibility
SAMPLE_OUTPUT = "sample_output.txt"
if not os.access(SAMPLE_OUTPUT, os.F_OK):
    raise Exception(('Error! Could not find a "{0}" file. '
                     'Make sure that there is a \"{0}\" in the same '
                     'directory as "{1}"! Spelling is very '
                     'important here.').format(SAMPLE_OUTPUT, __file__))
if not os.access(SAMPLE_OUTPUT, os.R_OK):
    raise Exception(('Error! Could not read the "{0}" file. '
                     'Make sure that it readable by changing its '
                     'permissions. You may need to get help from '
                     'your instructor.').format(SAMPLE_OUTPUT, __file__))
with open("sample_output.txt", 'rb') as output_file:
    hashed = hashlib.md5(output_file.read()).hexdigest()
if hashed != SAMPLE_OUTPUT_HASH:
    raise Exception(('Error! Hash mismatch. '
                     'Make sure that you did not modify your "{0}" file. You '
                     'may want to redownload the file.').format(SAMPLE_OUTPUT))
# Read in output file data
with open(SAMPLE_OUTPUT) as output_file:
    output = output_file.read()
sections = output.split("#" * 80 + "\n")
EXPECTED = {}
for section in sections[2:]:
    parameters, _, body = section.split("\n", maxsplit=2)
    name, loan, status = parameters[10:].strip().split(",")
    contents = re.split('(\* .* \**)', body)[1:]
    EXPECTED[(name, loan)] = {
        'print_introduction': (contents[0] + contents[1]),
        'input_name': (contents[2] + contents[3]),
        'print_rating': (contents[4] + contents[5]),
        'input_loan_amount': (contents[6] + contents[7]),
        'print_loan_availability': (contents[8] + contents[9]),
        'print_conclusion': (contents[10] + contents[11]),
        'test_loan': status == 'True',
        'main': body
    }

student_main.calculate_rating = calculate_rating
calculate_rating("")
calculate_rating.r()

PRE_MSG = ('-' * 67) + "\n\nFeedback:\n"


def wrong_output(context, when):
    return ("Your {} function printed the wrong stuff.\n"
            "This happened when I tried the {}.\n "
            "Read the Output Diff above, checking the start of "
            "each line for the symbols +, -, and ?.\n"
            "Each of these symbols have a specific meaning:\n"
            "  + means a line you need to REMOVE\n"
            "  - means a line you need to ADD\n"
            "  ? means a line you need to CHANGE\n"
            "Also look for ^ symbols to indicate individual characters "
            "that you need to change.").format(context, when)


class TestBankingComponents(unittest.TestCase):
    maxDiff = None

    def test_no_globals(self):
        """ No using global variables. """
        with open("banking.py", "r") as student_code_file:
            student_code = student_code_file.read()
        if student_code.count("global") > 1:
            msg = ("You are using the word global in your program. "
                   "One of those times is in the code you were "
                   "given (Grindlehook's calculate_rating function). "
                   "However, the other time(s) it was in code or a comment you"
                   " wrote. Global variables are bad. They are just awful. "
                   "You may not use global variables to solve this problem. "
                   "In fact, avoid using them the rest of your life.")
            self.fail(PRE_MSG + msg)

    @skip_unless_callable('print_introduction')
    def test_print_introduction(self):
        ''' print_introduction '''
        for (name, loan), output in EXPECTED.items():
            expected_output = output['print_introduction']
            with patch('sys.stdout', new=io.StringIO()) as captured_output:
                result = student_main.print_introduction()
            student_output = captured_output.getvalue()
            msg = wrong_output('`print_introduction`', 'name "{}"'.format(name))
            self.assertEqual(expected_output, student_output, msg=PRE_MSG + msg)
            msg = ("This function was not supposed to return anything. Instead"
                   " it returned {}.".format(repr(result)))
            self.assertIs(result, None, msg=PRE_MSG + msg)

    @skip_unless_callable('input_name')
    def test_input_name(self):
        ''' input_name '''
        for (name, loan), output in EXPECTED.items():
            expected_output = output['input_name']
            prompter = make_inputs(name)
            with patch('sys.stdout', new=io.StringIO()) as captured_output:
                with patch('builtins.input', prompter):
                    try:
                        result = student_main.input_name()
                    except StopIteration:
                        msg = ("Failed to accept input for name '{}'. "
                               "I typed '{}' but it wasn't accepted."
                               ).format(name)
                        self.fail(msg=PRE_MSG + msg)
            student_output = captured_output.getvalue()
            msg = wrong_output('`input_name`', 'name "{}"'.format(name))
            self.assertEqual(expected_output, student_output, msg=PRE_MSG + msg)
            msg = ("This function was supposed to return the user's name."
                   " Instead it returned {}.".format(repr(result)))
            self.assertEqual(result, name, msg=PRE_MSG + msg)

    @skip_unless_callable('input_loan_amount')
    def test_input_loan_amount(self):
        ''' input_loan_amount '''
        for (name, loan), output in EXPECTED.items():
            expected_output = output['input_loan_amount']
            prompter = make_inputs(loan)
            with patch('sys.stdout', new=io.StringIO()) as captured_output:
                with patch('builtins.input', prompter):
                    try:
                        result = student_main.input_loan_amount()
                    except StopIteration:
                        msg = ("Failed to accept input for loan amount '{}'. "
                               "I typed '{}' but it wasn't accepted."
                               ).format(name)
                        self.fail(msg=PRE_MSG + msg)
            student_output = captured_output.getvalue()
            msg = wrong_output('`input_loan_amount`', 'loan "{}"'.format(loan))
            self.assertEqual(expected_output, student_output, msg=PRE_MSG + msg)
            msg = ("This function was supposed to return the loan amount."
                   " Instead it returned {}.".format(repr(result)))
            self.assertEqual(result, int(loan), msg=PRE_MSG + msg)

    @skip_unless_callable('print_rating')
    def test_print_rating(self):
        ''' print_rating '''
        for (name, loan), output in EXPECTED.items():
            expected_output = output['print_rating']
            with patch('sys.stdout', new=io.StringIO()) as captured_output:
                calculate_rating.r()
                rating = calculate_rating(name);
                calculate_rating.r()
                result = student_main.print_rating(rating)
            student_output = captured_output.getvalue()
            msg = wrong_output('`print_rating`', 'name "{}"'.format(name))
            self.assertEqual(expected_output, student_output, msg=PRE_MSG + msg)
            msg = ("This function was not supposed to return anything. Instead"
                   " it returned {}.".format(repr(result)))
            self.assertIs(result, None, msg=PRE_MSG + msg)

    @skip_unless_callable('test_loan')
    def test_test_loan(self):
        ''' test_loan '''
        for (name, loan), output in EXPECTED.items():
            expected_result = output['test_loan']
            expected_output = ''
            with patch('sys.stdout', new=io.StringIO()) as captured_output:
                rating = calculate_rating(name);
                calculate_rating.r()
                result = student_main.test_loan(rating, int(loan))
            student_output = captured_output.getvalue()
            msg = wrong_output('`test_loan`', 'name "{}" and loan {}'.format(name, loan))
            self.assertEqual(expected_output, student_output, msg=PRE_MSG + msg)
            msg = ("This function was supposed to return the loan "
                   "availability. For {} requesting {}, this should have been "
                   "{}. Instead, it returned {}."
                   ).format(name, loan, repr(expected_result), repr(result))
            self.assertEqual(result, expected_result, msg=PRE_MSG + msg)

    @skip_unless_callable('print_loan_availability')
    def test_print_loan_availability(self):
        ''' print_loan_availability '''
        for (name, loan), output in EXPECTED.items():
            expected_output = output['print_loan_availability']
            with patch('sys.stdout', new=io.StringIO()) as captured_output:
                calculate_rating.r()
                rating = calculate_rating(name);
                calculate_rating.r()
                result = student_main.print_loan_availability(rating, int(loan))
            student_output = captured_output.getvalue()
            msg = wrong_output('`print_loan_availability`', 'name "{}" and loan {}'.format(name, loan))
            self.assertEqual(expected_output, student_output, msg=PRE_MSG + msg)
            msg = ("This function was not supposed to return anything. Instead"
                   " it returned {}.".format(repr(result)))
            self.assertIs(result, None, msg=PRE_MSG + msg)

    @skip_unless_callable('print_conclusion')
    def test_print_conclusion(self):
        ''' print_conclusion '''
        for (name, loan), output in EXPECTED.items():
            expected_output = output['print_conclusion']
            with patch('sys.stdout', new=io.StringIO()) as captured_output:
                result = student_main.print_conclusion()
            student_output = captured_output.getvalue()
            msg = wrong_output('`print_conclusion`', 'name "{}"'.format(name))
            self.assertEqual(expected_output, student_output, msg=PRE_MSG + msg)
            msg = ("This function was not supposed to return anything. Instead"
                   " it returned {}.".format(repr(result)))
            self.assertIs(result, None, msg=PRE_MSG + msg)


class TestBankingMain(unittest.TestCase):
    maxDiff = None

    @skip_unless_callable('main')
    def test_main(self):
        ''' main '''
        for (name, loan), output in EXPECTED.items():
            expected_output = output['main']
            prompter = make_inputs(name, loan)
            with patch('sys.stdout', new=io.StringIO()) as captured_output:
                with patch('builtins.input', prompter):
                    try:
                        calculate_rating.r()
                        result = student_main.main()
                    except StopIteration:
                        msg = ("Failed to accept input when calling main. "
                               "I typed '{}' and then '{}' but it wasn't "
                               "accepted."
                               ).format(name, loan)
                        self.fail(msg=PRE_MSG + msg)
            student_output = captured_output.getvalue()
            msg = wrong_output('`main`', 'name "{}" and loan "{}"'.format(name, loan))
            self.assertEqual(expected_output, student_output, msg=PRE_MSG + msg)
            msg = ("This function was not supposed to return anything. Instead"
                   " it returned {}.".format(repr(result)))
            self.assertIs(result, None, msg=PRE_MSG + msg)


if __name__ == '__main__':
    '''
    Only run the main test suite if the component functions all pass.
    '''
    runner = unittest.TextTestRunner()
    phase1_suite = unittest.TestSuite()
    phase1_suite.addTest(unittest.makeSuite(TestBankingComponents))
    result = runner.run(phase1_suite)
    if result.wasSuccessful() and not result.skipped:
        phase2_suite = unittest.TestSuite()
        phase2_suite.addTest(unittest.makeSuite(TestBankingMain))
        result2 = runner.run(phase2_suite)
        if result2.wasSuccessful():
            sys.stderr.flush()
            print("")
            print("CONGRATULATIONS! Your solution passes all the unit tests.")
            print("Clean and organize your code, so that it looks nice.")
            print("Check the style guide for recommendations on organization.")
            print("Then, submit on Web-CAT.")
            print("Great work on writing all that code :)")
