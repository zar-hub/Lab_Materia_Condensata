import os
import numpy as np
import pandas as pd
from pprint import pprint
import sys

_DEBUG = False

# phyisical constants
h = 6.62607015e-34  # J / Hz
kb = 1.380649e-23   # J / K
c = 299792458       # m / s
hc = h * c

def plank(wl, T):
    '''
    Returns the Plank's law of radiadiation density
    by wavelenght.
    wl : the wavelentgh in meters
    T : temperature in K
    '''
    A = 2 * h * c ** 2
    print(f'wl type is {type(wl)}')
    
    expm1 = np.expm1(hc / (wl * kb * T))
    return A / (np.power(wl, 5) * expm1)

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")

def getFiles(path : str):
    '''Gets a list of the file names in the 
        specified folder relative to the cwd.
        Note that the folder path should be like 
        "folder\\subfolder" '''
    cwd = os.getcwd()
    full_path = os.path.join(cwd, path)
    files = os.scandir(full_path)
    # only return the files
    files = [f for f in files if f.is_file()]
    return files

def dfFormFiles(files : list, **opt):
    '''Generate pandas dataframe from a list of os.files,
        passes opt to the pd.read_csv. 
        RESETS THE INDEX!'''
    df = pd.DataFrame()
    
    if _DEBUG:
        print(files,opt,sep='\n')
    
    for i, f in enumerate(files):
        # temp df for handling the numbers 
        tmp = pd.read_csv(f.path, **opt)
        tmp['meta'] = f.name
        df = df._append(tmp)
    df = df.reset_index(drop=True)
    
    return df