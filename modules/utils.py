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
    files = [f for f in files if f.is_file() and f.name.endswith('.csv')]
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

def token_parser(tokens : str):
    '''Parses the token and returns the meta data'''
    meta = dict()
    tokens = tokens.split('_')
    meta['type'] = tokens[0]
    for token in tokens[1:]:
        if token.find('K') != -1 or token.find('mV') != -1:
            meta['temp'] = token
        elif token.find('ms') != -1:
            meta['integration time'] = token
        elif token.find('V') != -1:
            meta['PT tension'] = token
        elif token.find('Out') != -1:
            meta['output slit'] = token
        elif token.find('In') != -1:
            meta['input slit'] = token
        elif token.find('Hz') != -1:
            meta['lock freq'] = token
        elif token.find('2024') != -1:
            meta['date'] = token
        else:
            meta['id'] = token
    
    # tidy up the meta data
    try:
        meta['input slit'] = meta['input slit'][:-2]
        meta['output slit'] = meta['output slit'][:-3]
    except:
        pass
    
    return meta

def file_to_series(file : os.DirEntry, **opt):
    ''' Converts a file to a pandas series
        opt: 
            - pfname : print the file name
            - pmeta : print the meta data
    '''
    data = np.genfromtxt(file.path, skip_header=1, dtype=np.float32)
    
    # get the std
    data[:,2] = data[:,2:].std(axis=1, ddof=1)
    data = data[:,:3].T
    
    # get the meta data
    tokens = file.name.removesuffix('.csv')
    meta = token_parser(tokens)
    
    ser = pd.Series()
    ser['type'] = meta['type']
    ser['date'] = meta['date']
    ser['id'] = meta['id']
    ser['interval'] = [data[0,0], data[0,-1]]
    ser['wl'] = data[0]
    ser['mean'] = data[1]
    ser['std'] = data[2]
    ser['meta'] = str(meta)
    
    if opt.get('pfname', False):
        print(file.name)
    if opt.get('pmeta', False):
        pprint(meta)
    
    return ser