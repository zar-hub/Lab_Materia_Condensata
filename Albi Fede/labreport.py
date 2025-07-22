import marimo

__generated_with = "0.13.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import os
    import re
    import matplotlib.pyplot as plt

    cwd = os.getcwd()

    # settings
    path = os.path
    OUT_PATHS = {'figs_dir': './figures', 'results_dir': './results'}
    IN_PATHS = {'raw_dir' : 'rawdata'}
    RAW_DIRS = ['12_12_2024', '17_12_2024', '19_12_2024', '20_12_2024']
    FIG_SIZE = np.array([6.4, 4])
    SAVE_FIG = True
    FIG_FMT = 'pdf'
    TRANSPARENT_PNG = True

    mo.md('''
    # Lab Report
    In questo laboratorio facciamo un analisi HREELS di un campione di Argento 111 su cui vengono
    depositate delle molecole.
    Abbiamo raccolto le misure dalle date {}
    '''.format(RAW_DIRS))
    return FIG_SIZE, cwd, mo, np, path, pd, plt, re


@app.cell
def _(cwd, path, pd):
    df = pd.read_csv(path.join(cwd, 'rawdata', 'dataframe'))
    df
    return (df,)


@app.cell
def _(mo):
    number_sld = mo.ui.slider(0, 30, 1, value=30)
    contrast_sld = mo.ui.slider(0, 100, 1, value=10)
    run_btn = mo.ui.run_button()
    return


@app.cell
def _(df, np, pd, re):
    def parse_spectre(spectre):
        """
        Parse the spectre string into a list of floats.
        """
        if not (type(spectre) is str):
            return np.array([[],[]])
        # remove the brackets and split by comma
        spectre = re.sub(r"[\n\],']", '', spectre)
        items = spectre.split('[')
        a = np.fromstring(items[2], sep=' ')
        b = np.fromstring(items[3], sep=' ')
        return a,b

    # unpack the spectre in x and wave fields
    clean_df = df.copy().drop('Unnamed: 0', axis=1)
    clean_df['spectre'] = clean_df['spectre'].apply(parse_spectre)
    clean_df[['x', 'wave']] = clean_df['spectre'].apply(lambda x : pd.Series([*x], index=['x', 'wave']))
    clean_df = clean_df.drop(columns = 'spectre')

    # keep only the data where spectre is longer than 10 elements...
    # it means that the scan did not end early
    # filter for number of elements and contrast
    # clean_df = clean_df[clean_df['x'].apply(lambda x: len(x) > number_sld.value)]
    # clean_df = clean_df[clean_df['wave'].apply(lambda x: x.max() / x.mean() > contrast_sld.value)]

    # parse the info of the dataframe from the meta field
    from ast import literal_eval
    def parsemeta(ser : pd.Series):
        assert type(ser) is pd.Series, ser
        meta = ser['meta']
        assert type(meta) is str, str(ser.head())
        meta = literal_eval(meta) 
        assert(type(meta) is dict )
        info = pd.Series(meta)

        # drop the old meta field and return 
        ser = ser.drop('meta')
        return pd.concat([ser, info])

    clean_df = clean_df.apply(parsemeta, axis = 1)
    clean_df.loc[:, 'size'] = clean_df['x'].apply(len)

    # remove items where size > 90
    clean_df = clean_df.query('size > 90')

    # filter for constrast
    def contrast(x : np.ndarray):
        assert type(x) is np.ndarray

        return x.max() / x.mean()

    def contrast_filter(ser: pd.Series):
        assert type(ser) is pd.Series

        cont = contrast(ser['wave'])
        # for IPRPCTDI the filter needs
        # to be adjusted
        if ser['sample'] == 'IPRPTCDI':
            res = (cont > 10) | (cont < 2)

        else:
            res = cont > 7.5
        print(ser['sample'], cont, res)
        return res

    clean_df = clean_df[clean_df.apply(contrast_filter, axis=1)]

    # _info = mo.md(f'There are {clean_df.shape[0]} instances where spectre has more than {number_sld.value} elements')
    # _info_contrast = mo.md(f'Keep the spectre where the max/mean ratio is greater than {contrast_sld.value}')


    # mo.vstack([mo.hstack([_info, number_sld]),
    #            mo.hstack([_info_contrast, contrast_sld]),
    #            clean_df, ])
    clean_df.drop(columns = ['x', 'wave'])
    return (clean_df,)


@app.cell
def _(clean_df):
    from igorwriter import IgorWave 
    # Print and Save the clean df


    def save_row(s):
        name = '_'.join([s['sample'], s['date'], 'R' + str(s['run'])])
        print(name)
        wave = IgorWave(s['wave'], name=name)
        wave.set_datascale('conteggi') 
        wave.set_dimscale('x', s['V_start_[V]'], s['V_step_[V]'], 'eV')
        wave.save_itx(name + '.itx')


    clean_df.apply(save_row, axis=1)
    clean_df
    return


@app.cell
def _(FIG_SIZE, clean_df, plt):
    figs = []

    for sample in ['Ag111', 'PTCDI', 'IPRPTCDI']:
        _s = clean_df.query('sample == @sample')

        _n = _s.shape[0]
        _n_rows = int(_n / 3) + 1
        _fig, _axs = plt.subplots(_n_rows, 3, figsize=FIG_SIZE * [1,_n_rows/2], dpi=300)
        _axs = _axs.flatten()
        for i in range(_n):
            _item = _s.iloc[i]
            _ax = _axs[i]
            _x, _y = _item['x'], _item['wave']
            _ax.plot(_x, _y)
            _ax.set_title(f'{_item['sample']}-{_item['run']}\n{_item['date']}')
            # _ax.set_yscale('log')
            _contrast = _y.max() / _y.mean()
            _ax.text(0.01, 0.93, f"cont : {_contrast : .2f}", transform=_ax.transAxes)

        _fig.tight_layout()
        _fig.subplots_adjust(wspace = 0.6)
        figs.append(_fig)

    tab1 = figs[0]
    tab2 = figs[1]
    tab3 = figs[2]
    return tab1, tab2, tab3


@app.cell
def _(mo, tab1, tab2, tab3):
    mo.ui.tabs({
        'Ag111':tab1,
        'PTCDI':tab2,
        'IPRPTCDI':tab3})
    return


@app.cell
def _(clean_df, mo, plt):
    # find the peaks on a sample data
    _mask = clean_df['sample'] == 'PTCDI'
    _d = clean_df[_mask]

    # we calibrate the spectra using
    # the data inside the range [-5.24, -4.9]
    _calib = [-5.24, -4.8]

    def keepinside(data, rng):
        '''
        For specific use on spectre data 
        in this notebook
        '''
        x, y = data[['x', 'wave']]
        left, right = rng
        mask = (x > left) & (x < right)
        x = x[mask]
        y = y[mask]
        data[['x', 'wave']] = x,y
        return data

    _slice = _d.apply(keepinside, rng = _calib, axis = 1)

    def pltall(df):
        fig, axs = plt.subplots(1,3, figsize=(15,5))

        for i, item in df.iterrows():
            x, y = item[['x', 'wave']]
            bias = (y.mean() + y.max()) / 2
            _bias = y.max()
            __bias = y.mean()
            axs[0].plot(x, y / bias)
            axs[1].plot(x, y / _bias)
            axs[2].plot(x, y / __bias)

        axs[0].set_title('mean + max')
        axs[1].set_title('max')
        axs[2].set_title('mean')
        return fig

    mo.vstack([
    mo.md('''
    Possiamo vedere dal grafico che la scelta migliore per la normalizzazione delle misure 
    è usare la media. In questo modo si ha la minor disperione sia attorno al picco che attorno alle code. Si potrebbero usare modi più sofisticati per ricostruire i dati, ma sarebbero più difficili da giustificare da un punto di vista fisico.
    '''),
    pltall(_slice)
    ])
    return


@app.cell
def _(clean_df, mo, plt):
    def norm_inrange(data, rng, x = None, y = None):
        '''
        Returns a list of normalization factors,
        one for each row.
        The normalization is done on the mean over 
        the given interval.
        '''

        # get the normalization from 
        # making the mean over the selected range

        x,y = data[['x', 'wave']]

        left, right = rng
        mask = (x > left) & (x < right)
        y = y[mask]
        norm = y.mean()

        # update the data
        y = data['wave'] / norm
        data[['x', 'wave']] = x, y
        return data 

    def plotSpectre(df):
        fig, ax = plt.subplots()
        for i, e in df.T.iterrows():
            label = f'{e['date']}-{e['run']}'
            x, y = e[['x', 'wave']]
            ax.plot(x, y, label = label)
        ax.set_xlim(-5.6, -5)
        ax.legend()
        return fig

    _ptcdi = clean_df[clean_df['sample'] == 'PTCDI']
    _calib = [-5.24, -4.8]
    _ptcdi = _ptcdi.T.apply(norm_inrange, rng = _calib)
    _fig = plotSpectre(_ptcdi)

    mo.vstack([
        mo.md('''
        Possiamo usare la normalizzazione nell'intervallo scelto per tutte le misure.
        Ci sono due misure che non sembrano essere normalizzate correttamente. 
        La soluzione è normalizzare le misure che partono da -5.7 separatamente e poi 
        unirle con le altre.
        Alla fine faremo la media di tutte le misure per ottenere un unico spettro.
        '''),
        _fig,
        _ptcdi.T
    ])
    return


@app.cell
def _(clean_df, np, pd):
    # 1) resample data to common x
    # 2) minimize loss function
    import scipy
    import plotly.express as px
    import plotly.graph_objects as go

    def interpWave(ser : pd.Series, commonx : np.ndarray):
        '''
        Generate a new list of interpolated data.
        Outside the bounds the new wave is zero.
        This will be useful to compute a mean only
        in overlapping regions of multiple waves.
        '''
        assert(type(ser) is pd.Series)
        assert(type(commonx) is np.ndarray) 

        y = np.interp(commonx, ser['x'], ser['wave'], left=-1, right=-1)
        return y

    ag111 = clean_df[clean_df['sample'] == 'Ag111']
    ptcdi = clean_df[clean_df['sample'] == 'PTCDI']
    iprptcdi = clean_df[clean_df['sample'] == 'IPRPTCDI']

    _min, _max = [], []
    for _df in ag111, ptcdi,iprptcdi:
        _min.append(_df['x'].map(min).min())
        _max.append(_df['x'].map(max).max())
    _min = min(_min) 
    _max = max(_max) 
    commonx = np.linspace(_min, _max, 700)



    # for _name, _df in interp_tabs.items():
    #     _fig, _ax = plt.subplots()
    #     _df.loc[:,'interpwave'] = _df.apply(interpWave, commonx = commonx, axis = 1)
    #     _df.apply(lambda x : 
    #               _ax.plot(commonx, x['interpwave'], 
    #               label = f"{x['date']}-{x['run']}"),
    #               axis = 1)


    #     _ax.set_xlim(-5.7, -4.7)
    #     _ax.legend()
    #     interp_tabs[_name] = _fig 



    return ag111, commonx, go, iprptcdi, ptcdi


@app.cell
def _(ag111, go, iprptcdi, mo, np, ptcdi):

    def pydfrow(s, fig):
        fig.add_trace(
            go.Scatter(
                x = s['x'],
                y = s['wave'],
                mode = 'lines',
                name = str(s['date']),
                customdata=np.column_stack([s['x'], s['wave']])
            )
        )

    fig1 = go.Figure()
    ag111.apply(pydfrow, fig = fig1, axis = 1)
    fig2 = go.Figure()
    ptcdi.apply(pydfrow, fig = fig2, axis = 1)
    fig3 = go.Figure()
    iprptcdi.apply(pydfrow, fig =fig3, axis = 1)

    for f in fig1, fig2, fig3:
        f.update_layout(
            title='Waveforms Colored by Date',
            xaxis_title='X Axis',
            yaxis_title='Counts',
            xaxis=dict(range=[-5.7, -4.8]),
            legend_title='Date',
        )
    plt1 = mo.ui.plotly(fig1)
    plt2 = mo.ui.plotly(fig2)
    plt3 = mo.ui.plotly(fig3)
    return plt1, plt2, plt3


@app.cell
def _(mo, plt1, plt2, plt3):
    interp_tabs = {
        'Ag111' : plt1,
        'PTCDI' : plt2,
        'IPRPTCDI' : plt3
    }

    import plotly

    mo.vstack([
        plt1,
        plt1.value
    ])
    return


@app.cell
def _():
    import altair as alt
    from vega_datasets import data

    source = data.stocks()

    alt.Chart(source).mark_line(interpolate="monotone").encode(
        x="date:T",
        y="price:Q",
        color="symbol:N"
    )
    return


@app.cell
def _(commonx, np, pd, plt, ptcdi):
    from IPython.display import display
    # now we use the most resoved spectre as a reference to normalize all the others:
    _ref = ptcdi.apply(lambda x : x['wave'].max(), axis = 1)
    _ref = _ref[_ref == _ref.max()].index
    _first = ptcdi.loc[_ref] 

    # create an ordered dataset where the fist element
    # is the reference normalized.
    # Also drop 'wave' and keep only interpolated data
    interp_df = pd.concat([_first, ptcdi.drop(_ref)], ignore_index=True)
    interp_df = interp_df.drop(columns=['x', 'wave'])

    def meanoverlap(df):
        '''
        DATA MUST BE NORMALIZED IN THE SAME WAY! 
        '''
        y = df['interpwave']
        # count how many data we have
        counts = y.apply(lambda x: (x >= 0).astype(int))
        # but then -1 should be zero when summed... so 
        y = y.apply(lambda y : np.where(y < 0, 0, y))
        return y.sum() / counts.sum()

    def norm_interp(data, rng, x = None):
        '''
        Returns a list of normalization factors,
        one for each row.
        The normalization is done on the mean over 
        the given interval.
        '''

        # get the normalization from 
        # making the mean over the selected range

        y = data['interpwave']

        left, right = rng
        mask = (x > left) & (x < right)
        y = y[mask]
        norm = y.mean()

        # update the data
        y = data['interpwave'] / norm
        data['interpwave'] = y
        return data 

    # normalization must be done first
    _calib = [-5.24, -2]
    interp_df = interp_df.apply(norm_interp, x=commonx ,rng = _calib, axis = 1)
    plt.plot(commonx, meanoverlap(interp_df))
    interp_df['interpwave'].apply(lambda x : plt.plot(commonx, x, alpha = 0.25))
    # plt.xlim(0,200)
    plt.yscale('log')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
