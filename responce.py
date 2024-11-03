import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.optimize import *

# Black Body dataframe
BB_Exp_df = pd.read_csv('BB_Radiation_Exp')
BB_Exp_df.plot('wl', 'mean')
plt.show()

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


def fitPropConst(x, a):
    return a * x 


# extract the relevant cols to be more efficient in plots
wl = BB_Exp_df['wl'].to_numpy(dtype='float64')
mean = BB_Exp_df['mean'].to_numpy(dtype='float64')
theo = plank(BB_Exp_df['wl'].to_numpy(dtype='float64') * 1e-10, 1260)

start = 10
stop = 40
fig, ax = plt.subplots()
ax.plot(wl[start:stop], theo[start:stop] * 4e-11)
ax.plot(wl[start:stop], mean[start:stop])
plt.show()
plt.plot(wl, fitPropConst(theo, 10))
plt.plot(wl, theo)
# optimize the proportionality coeff
try:
    popt, pcov = curve_fit(fitPropConst, mean[start:stop], theo[start:stop], p0=[1e11], bounds=(1e10, 1e12), method='trf')
except RuntimeError as e:
    print(e)
popt[0] 
pcov
# plots
ax = BB_Exp_df.plot('wl', 'mean')
ax.plot(wl, theo * 4e-11)
ax.set_yscale('log')
plt.show()
plt.plot(wl, BB_Exp_df['mean'] / theo)
