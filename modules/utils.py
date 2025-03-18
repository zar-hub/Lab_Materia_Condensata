import os
import numpy as np
import pandas as pd
from pprint import pprint
import sys
from scipy.special import wofz
import matplotlib
from typing import Tuple

_DEBUG = False

# CONSTANTS (SI UNITS)
h = 6.62607015e-34  # J / Hz
kb = 1.380649e-23   # J / K
c = 299792458       # m / s
hc = h * c

# HELPER FUNCTIONS
# pandas helper functions, os path maniuplation
# and CLI helpers
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

# https://zhauniarovich.com/post/2022/2022-09-matplotlib-graphs-in-research-papers/
def check_paths(in_paths, out_paths):
    import os, shutil, itertools
    for pth_key in in_paths:
        pth = in_paths[pth_key]
        if not os.path.exists(pth):
            print(f'Path [{pth}] does not exist')
        if pth_key.endswith('_dir') and (not os.path.isdir(pth)):
            print(f'Path [{pth}] does not correspond to a directory!')

    for pth_key in out_paths:
        pth = out_paths[pth_key]
        if pth_key.endswith('_dir'):
            abs_path = os.path.abspath(pth)
        else:
            abs_path = os.path.abspath(os.path.dirname(pth))
        if not os.path.exists(abs_path):
            print(f'Creating path: [{abs_path}]')
            os.makedirs(abs_path)

def save_fig(
        fig: matplotlib.figure.Figure, 
        fig_name: str, 
        fig_dir: str, 
        fig_fmt: str,
        fig_size: Tuple[float, float] = [6.4, 4], 
        save: bool = True, 
        dpi: int = 300,
        transparent_png = True,
    ):
    """This procedure stores the generated matplotlib figure to the specified 
    directory with the specified name and format.

    Parameters
    ----------
    fig : [type]
        Matplotlib figure instance
    fig_name : str
        File name where the figure is saved
    fig_dir : str
        Path to the directory where the figure is saved
    fig_fmt : str
        Format of the figure, the format should be supported by matplotlib 
        (additional logic only for pdf and png formats)
    fig_size : Tuple[float, float]
        Size of the figure in inches, by default [6.4, 4] 
    save : bool, optional
        If the figure should be saved, by default True. Set it to False if you 
        do not want to override already produced figures.
    dpi : int, optional
        Dots per inch - the density for rasterized format (png), by default 300
    transparent_png : bool, optional
        If the background should be transparent for png, by default True
    """
    if not save:
        return
    
    fig.set_size_inches(fig_size, forward=False)
    fig_fmt = fig_fmt.lower()
    fig_dir = os.path.join(fig_dir, fig_fmt)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    pth = os.path.join(
        fig_dir,
        '{}.{}'.format(fig_name, fig_fmt.lower())
    )
    if fig_fmt == 'pdf':
        metadata={
            'Creator' : '',
            'Producer': '',
            'CreationDate': None
        }
        fig.savefig(pth, bbox_inches='tight', metadata=metadata)
    elif fig_fmt == 'png':
        alpha = 0 if transparent_png else 1
        axes = fig.get_axes()
        fig.patch.set_alpha(alpha)
        for ax in axes:
            ax.patch.set_alpha(alpha)
        fig.savefig(
            pth, 
            bbox_inches='tight',
            dpi=dpi,
        )
    else:
        try:
            fig.savefig(pth, bbox_inches='tight')
        except Exception as e:
            print("Cannot save figure: {}".format(e)) 

            
# IGOR VOIGT FUNCTION WRAPPING
# from CHATGPT but checked for correctness
def voigt_func(x, shape_factor):
    """
    Computes the Voigt function as the real part of the Faddeeva function.
    :param x: Input wave (numpy array)
    :param shape_factor: Determines the Gaussian-Lorentzian mix.
    :return: Voigt peak values.
    """
    return np.real(wofz(x + 1j * shape_factor))

def mpfx_voigt_peak(cw, xw):
    """
    Computes a Voigt peak function and fills yw with computed values.
    :param cw: Coefficient wave (array-like of length 4)
               cw[0]: Peak location
               cw[1]: Width-affecting factor
               cw[2]: Amplitude factor
               cw[3]: Shape factor (0 = Gaussian, Inf = Lorentzian, sqrt(ln(2)) = 50/50)
    :param xw: X values where the function is evaluated
    :return: 0 if successful, NaN if input is invalid
    """
    try:
        # Validate input
        cw = np.asarray(cw, dtype=np.float64)
        xw = np.asarray(xw, dtype=np.float64)
        if len(cw) != 4:
            return np.nan
        
        # Compute the Voigt peak values
        return cw[2] * voigt_func(cw[1] * (xw - cw[0]), cw[3])
        
    except Exception:
        return np.nan  # Return NaN on failure

# PHYSICS FUNCTIONS
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