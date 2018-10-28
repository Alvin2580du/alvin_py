'''
This file is used to test your implementation of `adventure.py` for
project 4 (Text-based adventure game).

You should not modify this file, because the instructor will be running
your code against the original version anyway.

@author: acbart
'''
__version__ = 8
import unittest
from unittest.mock import patch
import sys
import os
import io
import re
import textwrap
import traceback
import contextlib
import pprint
import copy
import random

random.seed(0)

TOLERANCE = 3
STUDENT_FILENAME = "adventure.py"
FINAL_MODE = False

# Ensure the user has created the main file
try:
    with patch('sys.stdout', new=io.StringIO()) as captured_output:
        with patch('builtins.input', return_value='0'):
            with patch('time.sleep', return_value=None):
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
        # print(msg)
        return unittest.skip(msg)
    # print(msg)
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


# Prevent certain code constructs
class ProjectRulesViolation(Exception):
    pass


# No cheap built-ins
def no_builtins_exception(name):
    def f(*args, **kwargs):
        raise ProjectRulesViolation('Error! You seem to have used a builtin function.')

    return f


def human_type(a_value):
    return {
        list: 'List',
        dict: 'Dict',
        int: 'Integer',
        str: 'String',
        float: 'Float',
        bool: 'Boolean'
    }.get(type(a_value), type(a_value))


class TestBase(unittest.TestCase):
    maxDiff = None

    def make_feedback(self, msg, **format_args):
        return ("\n\nFeedback:\n" +
                textwrap.indent(msg.format(function=self.shortDescription(),
                                           **format_args), '  '))

    def run_function(self, function, args=None, prompter=None):
        if prompter is None:
            prompter = make_inputs('')
        if args is None:
            args = []
        input_failed = False
        result = None
        with patch('sys.stdout', new=io.StringIO()) as captured_output:
            with patch('builtins.input', prompter):
                with patch('time.sleep', return_value=None):
                    try:
                        result = function(*args)
                    except StopIteration:
                        input_failed = True
        student_output = captured_output.getvalue()
        return result, student_output, input_failed


class KeyError_(KeyError):
    # def __init__(self, *args):
    #    self.args = args
    def __str__(self):
        return BaseException.__str__(self)


@contextlib.contextmanager
def improve_runtime_exception(function, **parameters):
    parameters = copy.deepcopy(parameters)
    try:
        yield
    except Exception as e:
        defs = ""
        for key, value in parameters.items():
            defs += "{} = {}\n".format(key, pprint.pformat(value, indent=2, compact=True))
        vs = ", ".join(parameters.keys())
        code = "{defs}{name}({vs})".format(name=function, defs=defs, vs=vs)
        code = textwrap.indent(code, "    ")
        if isinstance(e, KeyError):
            new_args = (repr(e.args[0]) + ".\n\nThe error above occurred when I ran:\n" + code,)
            new_except = KeyError_(*new_args)
            new_except.__cause__ = e.__cause__
            new_except.__traceback__ = e.__traceback__
            new_except.__context__ = e.__context__
            raise new_except from None
        else:
            e.args = (e.args[0] + ".\n\nThe error above occurred when I ran:\n" + code,)
            raise e


class TestInitial(TestBase):
    @skip_unless_callable('get_initial_state')
    def test_get_initial_state(self):
        ''' get_initial_state '''
        with improve_runtime_exception('get_initial_state'):
            result, output, _ = self.run_function(student_main.get_initial_state)
        # Returned dictionary
        msg = self.make_feedback("{function}() must return a Dictionary.\n"
                                 "Instead it has returned a {t}",
                                 t=human_type(result))
        self.assertIsInstance(result, dict, msg=msg)
        # Check 'game status' field
        msg = self.make_feedback("{function}() must have a key named 'game status'.")
        self.assertIn('game status', result, msg=msg)
        msg = self.make_feedback("{function}() 'game status' must be 'playing'.\n"
                                 "Instead it is '{game_status}'.",
                                 game_status=result['game status'])
        self.assertEqual(result['game status'], 'playing', msg=msg)
        # Check 'location' field
        msg = self.make_feedback("{function}() must have a key named 'location'.")
        self.assertIn('location', result, msg=msg)
        msg = self.make_feedback("{function}() 'location' must be a String.\n"
                                 "Instead it is a {t}.",
                                 t=human_type(result['location']))
        self.assertIsInstance(result['location'], str, msg=msg)
        # Check minimum number of fields
        msg = self.make_feedback("{function}() must return a Dictionary with at least 4 keys.\n"
                                 "Instead it has {length}:\n  {keys}",
                                 length=len(result.keys()),
                                 keys="\n  ".join(map(repr, result.keys())))
        self.assertGreaterEqual(len(result.keys()), 4, msg=msg)

    @skip_unless_callable('print_introduction')
    def test_print_introduction(self):
        ''' print_introduction '''
        with improve_runtime_exception('print_introduction'):
            result, output, _ = self.run_function(student_main.print_introduction)
        # Check that something was printed
        msg = self.make_feedback("{function}() must print something.")
        self.assertNotEqual(output, "", msg=msg)


class TestComponents(TestBase):
    def setUp(self):
        self.player, output, _ = self.run_function(student_main.get_initial_state)

    def tearDown(self):
        self.player = None

    @skip_unless_callable('print_current_state')
    def test_print_current_state(self):
        ''' print_current_state '''

        with improve_runtime_exception('print_current_state', player=self.player):
            result, output, _ = self.run_function(student_main.print_current_state,
                                                  args=[self.player])
        # Check that something was printed
        msg = self.make_feedback("{function}(get_initial_state()) must print something.")
        self.assertNotEqual(output, "", msg=msg)
        # Check that the location was printed
        msg = self.make_feedback("{function}(get_initial_state()) must print the player's location."
                                 "\nI expected to see the string {location!r}."
                                 "\nBelow you can see what I found instead:\n" +
                                 textwrap.indent(output, ' |  '),
                                 location=self.player['location'])
        self.assertIn(self.player['location'], output, msg=msg)

    @skip_unless_callable('get_options')
    def test_get_options(self):
        ''' get_options '''
        with improve_runtime_exception('get_options', player=self.player):
            result, output, _ = self.run_function(student_main.get_options,
                                                  args=[self.player])
        # Check that a list was returned
        msg = self.make_feedback("{function}() must return a List.\n"
                                 "Instead it has returned a {t}",
                                 t=human_type(result))
        self.assertIsInstance(result, list, msg=msg)
        # Check that it contains strings
        msg = self.make_feedback("{function}() must return a non-empty List.\n"
                                 "Instead the list was empty")
        self.assertTrue(result, msg=msg)
        msg = self.make_feedback("{function}() must return a List of Strings.\n"
                                 "Instead it returned {t!r}.",
                                 t=result)
        for index, item in enumerate(result):
            self.assertIsInstance(item, str, msg=msg)

    @skip_unless_callable('print_options')
    def test_print_options(self):
        ''' print_options '''
        command_tests = [['home', 'second door', 'upstairs'],
                         ['open the door'],
                         ['attack', 'defend']]
        for command_test in command_tests:
            with improve_runtime_exception('print_options', options=command_test):
                result, output, _ = self.run_function(student_main.print_options,
                                                      args=[command_test])
            for command in command_test:
                # Assert that the command is as expected
                msg = self.make_feedback("{function}({args!r})\n"
                                         "Expected it to print {command!r} somewhere.\n"
                                         "\nBelow you can see what I found instead:\n" +
                                         textwrap.indent(output, ' |  '),
                                         args=command_test,
                                         command=command)
                self.assertIn(command.lower(), output.lower(), msg=msg)

    @skip_unless_callable('get_user_input')
    def test_get_user_input(self):
        ''' get_user_input '''
        # Try reasonable, valid options
        command_tests = [([], ['home', 'second door', 'upstairs'], 'home'),
                         ([], ['open the door'], 'open the door'),
                         (['hide', 'run'], ['attack', 'defend'], 'attack'),
                         (['wait', 'act', 'item'], ['mercy'], 'mercy'),
                         (['quit'], ['home'], 'quit'),
                         (['wait', 'act', 'item', 'quit'], ['home'], 'quit'),
                         (['wait', 'quit'], ['mercy'], 'quit')]
        for bad_commands, command_test, expected in command_tests:
            input_sequence = bad_commands + [command_test[0]]
            inputs = make_inputs(*input_sequence)
            with improve_runtime_exception('get_user_input', options=command_test):
                result, output, inf = self.run_function(student_main.get_user_input,
                                                        args=[command_test],
                                                        prompter=inputs)
            # Input failure
            msg = self.make_feedback("{function}({args!r})\n"
                                     "Failed to terminate after the following valid inputs:\n"
                                     "  {i}",
                                     args=command_test,
                                     result=result,
                                     command=expected,
                                     i=', '.join(map(repr, input_sequence)))
            self.assertFalse(inf, msg=msg)
            # Assert that it is actually a string
            msg = self.make_feedback("{function}({args!r})\n"
                                     "I typed the inputs {i}.\n"
                                     "Expected it to return a string.\n"
                                     "Instead, it returned {result!r}.",
                                     args=command_test,
                                     result=result,
                                     i=', '.join(map(repr, input_sequence))
                                     )
            self.assertIsInstance(result, str, msg=msg)
            # Incorrect return value
            msg = self.make_feedback("{function}({args!r})\n"
                                     "I typed the inputs {i}.\n"
                                     "Expected it to return {command!r}.\n"
                                     "Instead, it returned {result!r}.",
                                     args=command_test,
                                     result=result,
                                     command=expected,
                                     i=', '.join(map(repr, input_sequence)))
            self.assertEqual(expected.lower().strip(),
                             result.lower().strip(), msg=msg)

    @skip_unless_callable('process_command')
    def test_process_command(self):
        ''' process_command '''
        with improve_runtime_exception('process_command', command='quit',
                                       player=self.player):
            result, output, _ = self.run_function(student_main.process_command,
                                                  args=['quit', self.player])
        # Don't return anything
        msg = self.make_feedback("{function}('quit', get_initial_state())\n"
                                 "Should not return anything.\n"
                                 "Instead, it returned {result!r}",
                                 result=result)
        self.assertEqual(result, None, msg=msg)
        # Successfully mutate player state
        msg = self.make_feedback("{function}('quit', get_initial_state())\n"
                                 "Should retain a key named 'game status'.\n"
                                 "But that key appears to be missing.")
        self.assertIn('game status', self.player, msg=msg)
        msg = self.make_feedback("{function}('quit', get_initial_state())\n"
                                 "Has a key named 'game status'.\n"
                                 "But that key's value is not a string.")
        self.assertIsInstance(self.player['game status'], str, msg=msg)
        msg = self.make_feedback("{function}('quit', get_initial_state())\n"
                                 "The value associated with the 'game status' key\n"
                                 "should now be 'quit'. Instead it is {actual!r}.",
                                 actual=self.player['game status'])
        self.assertEqual(self.player['game status'].lower().strip(), 'quit', msg=msg)

    @skip_unless_callable('print_game_ending')
    def test_print_game_ending(self):
        ''' print_game_ending '''
        with improve_runtime_exception('print_game_ending', player=self.player):
            result, output, _ = self.run_function(student_main.print_game_ending,
                                                  args=[self.player])
        # Check that something was printed
        msg = self.make_feedback("{function}(get_initial_state()) must print something.")
        self.assertNotEqual(output, "", msg=msg)


class TestPaths(TestBase):
    def check_path_variable(self, name):
        # Variable exists
        msg = self.make_feedback("You have not created a variable named `{name}`.", name=name)
        self.assertTrue(hasattr(student_main, name), msg=msg)
        NAME_VALUE = getattr(student_main, name)
        # Variable is a list
        msg = self.make_feedback("Your `{name}` does not have a list value.", name=name)
        self.assertIsInstance(NAME_VALUE, list, msg=msg)
        # Variable is a list of strings
        msg = self.make_feedback("Your `{name}` does not have only string values.",
                                 name=name)
        for value in NAME_VALUE:
            self.assertIsInstance(value, str, msg=msg)
        # Variable is a non-empty list of strings
        msg = self.make_feedback("Your `{name}` is empty. It must have at least one command.",
                                 name=name)
        self.assertTrue(NAME_VALUE, msg=msg)

    def test_win_path(self):
        ''' WIN_PATH '''
        self.check_path_variable("WIN_PATH")

    def test_lose_path(self):
        ''' LOSE_PATH '''
        self.check_path_variable("LOSE_PATH")


class TestFunctionality(TestBase):
    # def test_basic_functionality(self):
    #    self.assertTrue(True, msg="You get one free point!")

    def test_win_lose_quit(self):
        ''' Game Functionality (winning and losing) '''
        WIN_PATH = student_main.WIN_PATH
        won = self.execute_path(WIN_PATH)
        LOSE_PATH = student_main.LOSE_PATH
        lost = self.execute_path(LOSE_PATH)
        quited = self.execute_path(['quit'])
        combos = [("Win", "Lose", won, lost),
                  ("Win", "Quit", won, quited),
                  ("Lost", "Quit", lost, quited)]
        for cn1, cn2, c1, c2 in combos:
            msg = self.make_feedback("Your game displays the same message for both\n" +
                                     "the {} and {} paths:\n".format(cn1, cn2) +
                                     " {}:\n".format(cn1) +
                                     textwrap.indent(c1, ' |  ') +
                                     " {}:\n".format(cn2) +
                                     textwrap.indent(c2, ' |  '))
            self.assertNotEqual(c1, c2, msg=msg)

    def execute_path(self, path):
        path_choice = make_inputs(*path)
        # Print out an introduction
        with improve_runtime_exception('print_introduction'):
            _, output, inf0 = self.run_function(student_main.print_introduction)
        # make the initial state
        with improve_runtime_exception('get_initial_state'):
            the_player, output, inf1 = self.run_function(student_main.get_initial_state)
        # Check victory or defeat
        while the_player['game status'] == 'playing':
            # Give current state
            with improve_runtime_exception('print_current_state', player=the_player):
                _, output, inf2 = self.run_function(student_main.print_current_state,
                                                    args=[the_player])
            # Get options
            with improve_runtime_exception('get_options', player=the_player):
                available_options, output, inf3 = self.run_function(student_main.get_options,
                                                                    args=[the_player])
            # Give next options
            with improve_runtime_exception('print_options', options=available_options):
                _, output, inf4 = self.run_function(student_main.print_options,
                                                    args=[available_options])
            # Get Valid User Input
            with improve_runtime_exception('get_user_input', options=available_options):
                chosen_command, output, inf5 = self.run_function(student_main.get_user_input,
                                                                 args=[available_options],
                                                                 prompter=path_choice)
            # Process Comands and change state
            with improve_runtime_exception('process_command', command=chosen_command,
                                           player=the_player):
                _, output, inf6 = self.run_function(student_main.process_command,
                                                    args=[chosen_command, the_player])
            if any((inf2, inf3, inf4, inf5, inf6)):
                break
        # Give user message
        with improve_runtime_exception('print_game_ending', player=the_player):
            _, final_output, inf7 = self.run_function(student_main.print_game_ending,
                                                      args=[the_player])
        # any failures
        if any((inf0, inf1, inf2, inf3, inf4, inf5, inf6, inf7)):
            failing_function = ('print_introduction' if inf0
                                else 'get_initial_state' if inf1
            else 'print_current_state' if inf2
            else 'get_options' if inf3
            else 'print_options' if inf4
            else 'get_user_input' if inf5
            else 'process_command' if inf6
            else 'print_game_ending' if inf7
            else '*unknown?*')
            msg = self.make_feedback("Your game did not end when I " +
                                     "ran I entered the following " +
                                     "commands:\n  " +
                                     ', '.join(map(repr, path)) +
                                     "\nI ran out of commands on the {} function."
                                     .format(failing_function))
            self.fail(msg)
        return final_output

    def setUp(self):
        result, output, _ = self.run_function(student_main.get_initial_state)
        self.player = result

    def tearDown(self):
        self.player = None


_reject_traceback_file_pattern = re.compile(r'[./|C:\\]')


class TestSuccessHolder(unittest.TextTestResult):
    def __init__(self, stream, descriptions, verbosity):
        super(TestSuccessHolder, self).__init__(stream, descriptions, verbosity)
        self.successes = []

    def addSuccess(self, test):
        super(TestSuccessHolder, self).addSuccess(test)
        self.successes.append(test)

    @unittest.result.failfast
    def addError(self, test, err):
        # print(repr(self._exc_info_to_string(err, test)))
        # self.errors.append((test, self._exc_info_to_string(err, test)))
        self.errors.append((test, self._exc_info_to_string(err, test)))
        # *self.__populateWithError(
        self._mirrorOutput = True
        if self.showAll:
            self.stream.writeln("ERROR")
        elif self.dots:
            self.stream.write('E')
            self.stream.flush()

    def _is_relevant_tb_level(self, tb):
        current_directory = os.path.dirname(os.path.realpath(__file__))
        filename, lineno, function_name, _ = traceback.extract_tb(tb, limit=1)[0]
        if filename == __file__:
            return True
        if filename.startswith(current_directory):
            return False
        if filename.startswith('.'):
            return False
        if not os.path.isabs(filename):
            return False
        return True
        # return '__unittest' in tb.tb_frame.f_globals

    '''
    def _exc_info_to_string(self, err, test):
        """Converts a sys.exc_info()-style tuple of values into a string."""
        exctype, value, tb = err
        # Skip test runner traceback levels
        while tb and self._is_relevant_tb_level(tb):
            tb = tb.tb_next

        if exctype is test.failureException:
            # Skip assert*() traceback levels
            length = self._count_relevant_tb_levels(tb)
        else:
            length = None
        tb_e = traceback.TracebackException(
            exctype, value, tb, limit=length, capture_locals=self.tb_locals)
        print(repr(exctype))
        sys.stdout.flush()
        msgLines = list(tb_e.format())

        if self.buffer:
            output = sys.stdout.getvalue()
            error = sys.stderr.getvalue()
            if output:
                if not output.endswith('\n'):
                    output += '\n'
                msgLines.append(unittest.result.STDOUT_LINE % output)
            if error:
                if not error.endswith('\n'):
                    error += '\n'
                msgLines.append(unittest.result.STDERR_LINE % error)
        return ''.join(msgLines)'''


UNIT_TEST_TYPES = [('successes', 'Success!'),
                   ('skipped', 'Skipped (function not defined)'),
                   ('failures', 'Test failed'),
                   ('errors', 'Test error (your code has an error!)'),
                   ('unexpectedSuccesses', 'Unexpected success'),
                   ('expectedFailures', 'Expected failure')]

if __name__ == '__main__':
    '''
    Only run the main test suite if the component functions all pass.
    '''
    runner = unittest.TextTestRunner(resultclass=TestSuccessHolder)
    phases = [("Initial", TestInitial),
              ("Components (Do the functions work?)", TestComponents),
              ("Paths (Did you define the path variables?)", TestPaths),
              ("Functionality (Can I play your game?)", TestFunctionality)]
    success = True
    total_cases = 0
    for number, (name, phase) in enumerate(phases):
        if not FINAL_MODE:
            print("#" * 70)
            print("Testing Phase {}: {}".format(number, name))
            print("Summary: ", end="")
        sys.stderr.flush()
        sys.stdout.flush()
        phase_suite = unittest.TestSuite()
        phase_suite.addTest(unittest.makeSuite(phase))
        total_cases += phase_suite.countTestCases()
        result = runner.run(phase_suite)
        for UNIT_TEST_TYPE, MESSAGE in UNIT_TEST_TYPES:
            for case in getattr(result, UNIT_TEST_TYPE):
                if isinstance(case, tuple):
                    case = case[0]
                print("\t", case.shortDescription() + ":", MESSAGE)
        success = success and (result.wasSuccessful() and not result.skipped)
        if not FINAL_MODE and not success:
            break
    sys.stderr.flush()
    sys.stdout.flush()
    if success:
        print("")
        print("CONGRATULATIONS! Your solution passes all the unit tests.")
        print("Clean and organize your code, so that it looks nice.")
        print("Check the style guide for recommendations on organization.")
        print("Please make sure your game is at least a little creative.")
        print("Then, submit on Web-CAT.")
        print("Great work on writing all that code :)")
