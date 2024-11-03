import os
import pandas as pd
from pprint import pprint

_DEBUG = False

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