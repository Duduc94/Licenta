"""
Author: Duduc Ionut
Date: 15.05.2017
"""
import os
import shutil


def ensure_dir(path):
    if not os.path.exists(path + '\\unlabeled'):
        return os.makedirs(path + '\\unlabeled')


def clear_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)
