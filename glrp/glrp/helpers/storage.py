import os
import sys
import glob


def create_dirs(dirnames):
    '''create logging directory for logging'''
    try:
        for dir_ in dirnames:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Could not create logging directories: {0}".format(err))
        sys.exit(-1)
