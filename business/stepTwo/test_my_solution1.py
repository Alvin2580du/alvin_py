'''
This file is used to test your implementation of `stats.py` for
project 3 (Survey Statistics).

You should not modify this file, because the instructor will be running
your code against the original version anyway.

@author: acbart
'''
__version__ = 1
import unittest
from unittest.mock import patch
import sys
import os
import io
import re
import ast
import hashlib

TOLERANCE = 2
SAMPLE_OUTPUT = "sample_results.txt"
STUDENT_FILENAME = "stats.py"
SAMPLE_OUTPUT_HASH = "0fc274779ad2e017356e877a67c71d41"
FINAL_MODE = False

# Ensure the user has created the main file
try:
    with patch('sys.stdout', new=io.StringIO()) as captured_output:
        with patch('builtins.input', return_value='0'):
            student_main = __import__(STUDENT_FILENAME[:-3])
except ImportError:
    raise Exception(('Error! Could not find a "{1}" file. '
                     'Make sure that there is a "{1}" in the same '
                     'directory as "{0}"! Spelling is very '
                     'important here.').format(__file__, STUDENT_FILENAME))

# Make sure we are using the right Python version.
PYTHON_3 = (sys.version_info >= (3, 0))
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
    if FINAL_MODE:
        return lambda func: func
    msg = ("You have not defined `{0}` in {1} as a function."
           ).format(function_name, STUDENT_FILENAME)
    if hasattr(student_main, function_name):
        if callable(getattr(student_main, function_name)):
            return lambda func: func
        print(msg)
        return unittest.skip(msg)
    print(msg)
    return unittest.skip(msg)


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


# Verify output file's integrity and accessibility
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
# Check the sample_output's contents are correct
with open(SAMPLE_OUTPUT, 'rb') as output_file:
    hashed = hashlib.md5(output_file.read()).hexdigest()
if hashed != SAMPLE_OUTPUT_HASH:
    raise Exception(('Error! Hash mismatch. '
                     'Make sure that you did not modify your "{0}" file. You '
                     'may want to redownload the file. '
                     'New hash:{1}').format(SAMPLE_OUTPUT, hashed))

# Read in output file data
EXPECTED = {}
with open(SAMPLE_OUTPUT) as output_file:
    function_name = 'No function specified'
    for line in output_file:
        if line.startswith("\t"):
            parse = lambda v: ast.literal_eval(v.strip())
            argument, result = map(parse, line.strip().split(":"))
            EXPECTED[function_name].append((argument, result))
        else:
            function_name = line.strip()
            EXPECTED[function_name] = []


# Prevent certain code constructs
class ProjectRulesViolation(Exception):
    pass


# No cheap built-ins
def no_builtins_exception(name):
    def f(*args, **kwargs):
        raise ProjectRulesViolation('Error! You seem to have used a builtin function.')

    return f


FORBIDDEN_BUILTINS = {
    'sum': no_builtins_exception('sum'),
    'len': no_builtins_exception('len'),
    'max': no_builtins_exception('max'),
    'min': no_builtins_exception('min'),
    'eval': no_builtins_exception('eval'),
    'exec': no_builtins_exception('exec'),
}

# No importing modules
with open(STUDENT_FILENAME, 'rb') as output_file:
    student_ast = ast.parse(output_file.read())
for statement in ast.walk(student_ast):
    if isinstance(statement, ast.Import):
        raise ProjectRulesViolation("You have used an import statement.")
    if isinstance(statement, ast.Call):
        if isinstance(statement.func, ast.Name):
            if statement.func.id == '__import__':
                raise ProjectRulesViolation("You have called __import__.")


class TestComponents(unittest.TestCase):
    maxDiff = None

    def wrong_result(self, arg, expected, actual):
        msg = ("\nIncorrect value returned for `{0}` function.\n"
               "\tGiven argument: {1}\n"
               "\tExpected result: {2}\n"
               "\tActual result: {3}\n"
               ).format(self.shortDescription(), arg, expected, actual)
        self.assertAlmostEqual(expected, actual, places=TOLERANCE, msg=msg)

    def generic_test(self, function):
        for args, expected_result in EXPECTED[self.shortDescription()]:
            with patch.multiple('builtins', **FORBIDDEN_BUILTINS):
                actual_result = function(args)
            self.wrong_result(args, expected_result, actual_result)

    @skip_unless_callable('count')
    def test_count(self):
        ''' count '''
        self.generic_test(student_main.count)

    @skip_unless_callable('summate')
    def test_summate(self):
        ''' summate '''
        self.generic_test(student_main.summate)

    @skip_unless_callable('mean')
    def test_mean(self):
        ''' mean '''
        self.generic_test(student_main.mean)

    @skip_unless_callable('maximum')
    def test_maximum(self):
        ''' maximum '''
        self.generic_test(student_main.maximum)

    @skip_unless_callable('minimum')
    def test_minimum(self):
        ''' minimum '''
        self.generic_test(student_main.minimum)

    @skip_unless_callable('median')
    def test_median(self):
        ''' median '''
        self.generic_test(student_main.median)

    @skip_unless_callable('square')
    def test_square(self):
        ''' square '''
        self.generic_test(student_main.square)

    @skip_unless_callable('standard_deviation')
    def test_standard_deviation(self):
        ''' standard_deviation '''
        self.generic_test(student_main.standard_deviation)

    def test_survey_question(self):
        msg = ("You have not created a variable named `SURVEY_QUESTION`.")
        self.assertTrue(hasattr(student_main, 'SURVEY_QUESTION'), msg=msg)
        SURVEY_QUESTION = student_main.SURVEY_QUESTION
        msg = ("Your `SURVEY_QUESTION` does not have a string value.")
        self.assertIsInstance(SURVEY_QUESTION, str, msg=msg)

    def test_survey_results(self):
        msg = ("You have not created a variable named `SURVEY_RESULTS`.")
        self.assertTrue(hasattr(student_main, 'SURVEY_RESULTS'), msg=msg)
        SURVEY_RESULTS = student_main.SURVEY_RESULTS
        msg = ("Your `SURVEY_RESULTS` does not have a list value.")
        self.assertIsInstance(SURVEY_RESULTS, list, msg=msg)
        msg = ("Your `SURVEY_RESULTS` does not have only integers and "
               "float values inside of it.")
        for value in SURVEY_RESULTS:
            self.assertIsInstance(value, (int, float), msg=msg)
        msg = ("Your `SURVEY_RESULTS` has fewer than 10 responses "
               "inside of it.")
        self.assertGreaterEqual(len(SURVEY_RESULTS), 10, msg=msg)


if __name__ == '__main__':
    '''
    Only run the main test suite if the component functions all pass.
    '''
    runner = unittest.TextTestRunner()
    phase1_suite = unittest.TestSuite()
    phase1_suite.addTest(unittest.makeSuite(TestComponents))
    result = runner.run(phase1_suite)
    if result.wasSuccessful() and not result.skipped:
        sys.stderr.flush()
        print("")
        print("CONGRATULATIONS! Your solution passes all the unit tests.")
        print("Clean and organize your code, so that it looks nice.")
        print("Check the style guide for recommendations on organization.")
        print("Then, submit on Web-CAT.")
        print("Great work on writing all that code :)")
