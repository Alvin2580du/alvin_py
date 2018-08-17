"""
The CMU module provides access to the Carnegie Mellon twitter tokenizer. It is
used internally by other modules in the `sussex_nltk` package and should not be
called directly.

.. codeauthor::
    Matti Lyra
"""

import os
import subprocess
import platform
import tempfile

import nltk
from nltk.internals import config_java, java

import sussex_nltk as susx


_paths = [os.path.join('C:\\','Program Files (x86)','Java','jre7','bin','java.exe'),
    os.path.join('C:\\','Program Files','Java','jre7','bin','java.exe'),
    os.path.join('C:\\','Program Files (x86)','Java','jre6','bin','java.exe'),
    os.path.join('C:\\','Program Files','Java','jre6','bin','java.exe'),
    os.path.join('C:\\','Program Files (x86)','Java','jre1.8.0_92','bin','java.exe'),
    os.path.join('C:\\','Program Files','Java','jdk1.8.0_131','bin','java.exe')]


def tag(sents, java_options='-Xmx1g -XX:ParallelGCThreads=2'):
    """Tags a sentence using the CMU twitter tokenizer.

    :param sents: List of sentences to be tagged. The list should
        contain each sentence as a string.
    :type sents: list of str
    """

    _root = os.path.join(susx._sussex_root, 'CMU')
    _cp = ''


    jars = [os.path.join(_root,jar) for jar in os.listdir(_root) if jar.endswith('.jar')]
    _cp += ';'.join(jars)

    # write the sentences to the temp file
    _input_fh, _input_file_path = tempfile.mkstemp(text=True)
    _input_fh = os.fdopen(_input_fh, 'w')
    _input = '\n'.join(x.strip() for x in sents if x.strip())
    _input_fh.write(_input)
    _input_fh.close()

    _output_fh, _output_file_path = tempfile.mkstemp(text=True)
    # if we're on windows and java hasn't been configured yet
    if platform.platform().startswith('Windows'):
        if nltk.internals._java_bin is None:
            found_java = False
            for jre_path in _paths:
                if os.path.exists(jre_path):
                    found_java = True
                    break
            if found_java:
                config_java(jre_path, options=java_options, verbose=False)
            else:
                raise RuntimeError(
                    'Can\'t find an installed Java Runtime Environment (JRE).'
                    'If you have installed java in a non standard location '
                    'please call nltk.internals.config_java with the correct '
                    'JRE path and options=\'-Xmx1g -XX:ParallelGCThreads=2\' '
                    'before calling sussex_nltk.cmu.tag')
    else:
        config_java(options=java_options, verbose=False)

    _cmd = ['cmu.arktweetnlp.RunTagger', '--no-confidence', '--output-format', 'conll', _input_file_path]

    _dir = os.getcwd()
    os.chdir(_root)
    java(_cmd, classpath=_cp, stdout=_output_fh, stderr=subprocess.PIPE)
    os.chdir(_dir)

    _output_file = open(_output_file_path, 'r')
    _output_data = _output_file.read()
    _output_file.close()
    os.fdopen(_output_fh).close()
    os.unlink(_input_file_path)
    os.unlink(_output_file_path)

    return _output_data


def java(cmd, classpath=None, stdin=None, stdout=None, stderr=None,
         blocking=True):
    """
    Execute the given java command, by opening a subprocess that calls
    `java`.  If `java` has not yet been configured, it will be configured
    by calling `config_java()` with no arguments.

    :param cmd: The Java command that should be called,
        formatted as
        a list of strings.  Typically, the first string will be the name
        of the java class; and the remaining strings will be arguments
        for that java class.
    :type cmd: list of str

    :param classpath: A colon `:` separated list of directories, JAR
        archives, and ZIP archives to search for class files.
    :type classpath: str

    :param str stdin, stdout, stderr: Specify the executed programs'
        standard input, standard output and standard error file
        handles, respectively.  Valid values are `subprocess.PIPE`,
        an existing file descriptor (a positive integer), an existing
        file object, and `None`.  `subprocess.PIPE` indicates that a
        new pipe to the child should be created.  With `None`, no
        redirection will occur; the child's file handles will be
        inherited from the parent.  Additionally, `stderr` can be
        `subprocess.STDOUT`, which indicates that the `stderr` data
        from the applications should be captured into the same file
        handle as for `stdout`.

    :param bool blocking: If `False`, then return immediately after
        spawning the subprocess.  In this case, the return value is
        the `Popen` object, and not a `(stdout, stderr)` tuple.

    :return: If `blocking=True`, then return a tuple `(stdout,
        stderr)`, containing the `stdout` and `stderr` outputs generated
        by the `java` command if the `stdout` and `stderr` parameters
        were set to `subprocess.PIPE`; otherwise returns `None`.  If
        `blocking=False`, then return a `subprocess.Popen` object.

    :raise OSError: If the java command returns a nonzero return code.
    """
    if stdin == 'pipe': stdin = subprocess.PIPE
    if stdout == 'pipe': stdout = subprocess.PIPE
    if stderr == 'pipe': stderr = subprocess.PIPE
    if isinstance(cmd, str):
        raise TypeError('cmd should be a list of strings')

    # Make sure we know where a java binary is.
    if nltk.internals._java_bin is None:
        config_java()

    # Set up the classpath.
    if platform.platform().startswith('Windows'):
        _java_cp_sep = ';'
    else:
        _java_cp_sep = ':'

    # Construct the full command string.
    cmd = list(cmd)
    cmd = ['-cp', classpath] + cmd
    cmd = [nltk.internals._java_bin] + nltk.internals._java_options + cmd

    # Call java via a subprocess
    p = subprocess.Popen(cmd, stdin=stdin, stdout=stdout, stderr=stderr)
    if not blocking: return p
    (stdout, stderr) = p.communicate()

    # Check the return code.
    if p.returncode != 0:
        print(stderr)
        raise OSError('Java command failed!')

    return (stdout, stderr)
