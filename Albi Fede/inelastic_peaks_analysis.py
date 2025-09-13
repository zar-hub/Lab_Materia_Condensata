import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


types = {
	'energy' : np.float64,
	'area' : np.float64,
	'FWHM' : np.float64,
}

df = pd.read_csv('inelastic_peaks.csv', dtype = types)

# remove peaks coming from artefacts
df = df[df['energy'] > 0.008]
df = df[df['area'] > 0]
df = df[df['energy_sigma'] < 0.3]
df = df[df['energy_sigma'] > 0]

df['wave number'] = df['energy'] * 8065.73


with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)


res = pd.DataFrame()
# group by sample_id to get relative intensity
for name, group in df.groupby('sample_id'):
	print(f"Sample ID: {name}")

    # Calculate the absolute difference from the reference energy (0.4 eV)
    # Store this in a new variable to avoid overwriting the 'energy' Series
	abs_diff_to_reference = abs(group['energy'] - 0.4)
    
    # Find the index of the peak with the minimum absolute difference
	idx = abs_diff_to_reference.idxmin()

    # Retrieve the full row for the identified reference peak
    # Use .loc[idx] which correctly uses the index label returned by idxmin()
	reference_peak_row = group.loc[idx]

	# create the relative intesity
	group['RI'] = group['area'] / reference_peak_row['area'] * 100

	print(f"Index of the peak closest to 0.4 eV: {idx}")
	print(f"Absolute energy differences from 0.4 eV for all peaks in this group:\n{abs_diff_to_reference}")
	print(f"Details of the reference peak (closest to 0.4 eV):\n{reference_peak_row}")
	print(f"Relative intensity (closest to 0.4 eV):\n{group['RI']}")
	print("-" * 40) # Separator for better readability
	res = pd.concat([res, group])

res = res[['sample_id', 'peak_id', 'energy', 'energy_sigma', 'wave number', 'area', 'RI', 'FWHM']]
res = res.sort_values('energy')


# GRAPH STUFF
print('++++++++++++++++++++==')
res['type'] = res['sample_id'].apply(lambda x: x.split('_')[0])

for name, group in res.groupby('type'):
	plt.errorbar(group['energy'], group['RI'],
		xerr = group['energy_sigma'], label=name, alpha =0.5, lw=1, fmt='.')

plt.xlabel('Binding energy eV')
plt.ylabel('Relative intensity')
plt.legend()	
plt.show()

from math import log10, floor
def round_sig(x, sig=2):
	return round(x, sig-int(floor(log10(abs(x))))-1)
# SAVE
res['RI'] = res['RI'].astype(int)
res['energy'] = res['energy'].apply(lambda x : round_sig(x, 4))
res['energy_sigma'] = res['energy_sigma'].apply(lambda x : round_sig(x))
res['wave number'] = res['wave number'].apply(lambda x : round_sig(x, 4))
res['area'] = res['area'].apply(lambda x : round_sig(x, 4))
res['FWHM'] = res['FWHM'].apply(lambda x : round_sig(x, 3))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(res)
res.to_csv('peaks_results.csv')
	

