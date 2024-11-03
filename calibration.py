import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os  
import os.path as path

# set precision
# np.set_printoptions(2)
# Calibrate the BB Spectrum 
cwd = os.getcwd()
calib_path = path.join(cwd, 'BB_Radiation', 'misure_calibr')
BB_files = os.scandir(calib_path)
BB_files = [f for f in BB_files if f.is_file()]
calib_df = pd.DataFrame()

col_names = ['wl', 'mean', '1', '2', '3']
# add the data to the calib database
for i, f in enumerate(BB_files):
    # temp df for handling the numbers 
    tmp = pd.read_csv(path.join(calib_path, f),
                      sep=r'    ',
                      names=col_names,
                      engine='python',
                      dtype=np.float32)
    tmp['meta'] = f.name
    calib_df = calib_df._append(tmp)
calib_df.reset_index()

# list wiht first numbers
first = [ gr['wl'].values[0] for name, gr in calib_df.groupby(['meta']) ]
last = first.copy()
# the very last edge is manually setted
del last[0]
last.append(10035.)
# for some reason the there is a jump in the data
# between 6000 and 7000 A, so it's better to leave
# it as it is
last[1] = calib_df[calib_df['meta'] == 'BB_RAD_300ms_700V_10p5tac_303Hz__6mil7mil_241024']['wl'].values[-1]

def resample(df : pd.DataFrame, first, last):
    tmp = pd.DataFrame()
    for i, [name, gr] in enumerate(df.groupby('meta')):
        print(i, name, f"from {first[i]:,.2f} to {last[i]:,.2f}")
        arr = np.linspace(first[i], last[i], len(gr.index))
        gr['wl'] = arr
        gr['meta'] = name
        tmp = tmp._append(gr)
    tmp = tmp.reset_index(drop=True)
    return tmp

calib_df = resample(calib_df, first, last)
print(calib_df[col_names])
calib_df.plot('wl', 'mean')
plt.show()
calib_df.to_csv('BB_Radiation_Exp')
