# -*- coding: utf-8 -*-
import glob
import os


def get_file_name_and_ext(filename):
    (file_path, temp_filename) = os.path.split(filename)
    (file_name, file_ext) = os.path.splitext(temp_filename)
    return file_name, file_ext


def get_files(dir, file_type='*.*', recursive=True):
    all_files = []
    if dir:
        dir = dir.strip()
    if not os.path.isabs(dir):
        dir = os.path.abspath(dir)
    des_dir = os.path.join(dir, file_type)
    for file in glob.glob(des_dir):
        all_files.append(file)
    if recursive:
        sub_dirs = get_dirs(dir)
        for sub_dir in sub_dirs:
            sub_dir = os.path.join(sub_dir, file_type)
            for file in glob.glob(sub_dir):
                all_files.append(file)
    return sorted(all_files)


def get_dirs(dir):
    dirs = []
    for root_dir, sub_dirs, files in os.walk(dir):
        for sub_dir in sub_dirs:
            dirs.append(os.path.join(root_dir, sub_dir))
    return dirs


def get_parent_dir(file_path):
    file_path, file_name = os.path.split(file_path)
    _, parent_dir = os.path.split(file_path)
    return parent_dir


def get_sub_directory_name(dir):
    dirs = []
    for root_dir, sub_dirs, _ in os.walk(dir):
        for sub_dir in sub_dirs:
            dirs.append(sub_dir)
    return dirs