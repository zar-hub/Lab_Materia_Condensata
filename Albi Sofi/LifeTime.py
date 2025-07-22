import marimo

__generated_with = "0.11.26"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Lifetime of the excited state""")
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from modules import utils
    import pandas as pd
    import ast
    import scipy as sp
    from matplotlib import cm
    from functools import partial
    from typing import Literal

    OUT_PATHS = {
        'figs_dir' : './figures',
        'results_dir' : './results',
    }
    IN_PATHS = {
        'black_body_dir' : './raw_data/Black_Body_Radiation',
        'na_dir' : './raw_data/Na_Spectrum',
        'ruby_dir' : './raw_data/Ruby_Spectrum',
        'laser_dir' : './raw_data/Laser_Spectrum',
    }

    # default figure settings
    FIG_SIZE = np.array([6.4, 4]) # 16:10 aspect ratio
    SAVE_FIG = True
    FIG_FMT = 'pdf'
    TRANSPARENT_PNG=True

    # setting up stuff
    savefig = partial(utils.save_fig, fig_dir=OUT_PATHS['figs_dir'], fig_fmt=FIG_FMT, fig_size=FIG_SIZE, save=SAVE_FIG, transparent_png=TRANSPARENT_PNG)
    utils.check_paths(IN_PATHS, OUT_PATHS)

    plt.close('all')
    return (
        FIG_FMT,
        FIG_SIZE,
        IN_PATHS,
        Literal,
        OUT_PATHS,
        SAVE_FIG,
        TRANSPARENT_PNG,
        ast,
        cm,
        np,
        partial,
        pd,
        plt,
        savefig,
        sp,
        utils,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # Model creation
        Here we create the model of the fitting function and we make sure it works as intended
        """
    )
    return


@app.cell
def _(np, plt):
    def expFlipFlop(_t, _tau, _T):
        """ 
        t: time in [s]
        tau: lifetime [s]
        T: window of periodicity [s]

        note: T should be the FULL period of the 
        square wave with 50% duty cycle
        """
        Thalf = _T / 2
        tstar = np.mod(_t, Thalf)
        tprime = _tau * np.log(1 + np.exp(-Thalf / _tau))
        widownum = np.floor_divide(_t, Thalf)
        risingedg = np.remainder(widownum + 1, 2)
        fallingedg = np.remainder(widownum, 2)
        _A = 1 - np.exp(-(Thalf + tprime) / _tau)
        rising = 1 - np.exp(-(tstar + tprime) * risingedg / _tau)
        falling = _A * np.exp(-tstar * fallingedg / _tau)
        return rising * risingedg + falling * fallingedg

    def expFlipFlop2(t, tau, tnot, T, A):
        return A * expFlipFlop(t - tnot, tau, T)
    _t = np.linspace(0, 10, 100)
    _A = expFlipFlop(_t - 0.4, 2, 4)
    plt.plot(_t, _A)
    plt.ylim(0,1)
    plt.xlabel(r'time $[ms]$')
    plt.ylabel(r'intensity $[a.u.]$')
    return expFlipFlop, expFlipFlop2


@app.cell
def _(df, expFlipFlop2, plt):
    from lmfit.models import Model
    _batch = df.iloc[28]

    _t = _batch['t'] - _batch['t'][0]
    _A = -_batch['A']
    model = Model(expFlipFlop2)
    _params = model.make_params()
    _params['A'].set(0.5, min=0)
    _params['tau'].set(0.007, min=0)
    _params['T'].set(0.014, min=0)
    _params['tnot'].set(0.01)
    _result = model.fit(_A, _params, t=_t)
    print(_result.fit_report())
    plt.plot(_t, _A)
    plt.plot(_t + _result.params['T'] / 2, -_A + _A.max())
    # plt.plot(_t, model.eval(_params, t=_t))
    plt.plot(_t, _result.best_fit)
    plt.show()
    return Model, model


@app.cell
def _(mo):
    mo.md("""it seems like this is not symmetric, it means that the model is wrong, this is because ther is also stimulated emission!""")
    return


@app.cell
def _(df, model, np, plt):
    _batch = df.iloc[28]

    _t = _batch['t'] - _batch['t'][0]
    _A = -_batch['A']
    _params = model.make_params()
    _params['A'].set(0.5, min=0)
    _params['tau'].set(0.007, min=0)
    _params['T'].set(0.013, min=0)
    _params['tnot'].set(0.01)

    # we fit the other curve
    #_A = _A + -_A + _A.max()

    # combine the curves
    _shift = np.argmin( (_t - _t[0]) < 0.013 / 2 )
    print(_shift)
    _A = _A[: -_shift ] + (-_A + _A.max())[_shift : ]
    _t = _t[:-_shift]

    _result = model.fit(_A, _params, t=_t)
    print(_result.fit_report())
    plt.plot(_t, _A)
    #plt.plot(_t + _result.params['T'] / 2, -_A + _A.max())
    # plt.plot(_t, model.eval(_params, t=_t))
    plt.plot(_t, _result.best_fit)
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Data analysis
        The first thing to do is to prepare the dataframe
        """
    )
    return


@app.cell
def _(np, pd, utils):
    from modules.thermoc import thermocouple


    files = utils.getFiles('Raw_Data\\Ruby_Spectrum')
    files = [file for file in files if file.name.find('RbTR_') != -1]

    def TRtoSeries(file):
        data = np.loadtxt(file, skiprows=9, delimiter=',', usecols=(0, 1, 2)).T

         # get the meta data
        tokens = file.name.removesuffix('.csv')
        tokens = tokens[:-14]
        meta = utils.token_parser(tokens)

        ser = pd.Series()
        ser['type'] = meta['type']
        ser['date'] = meta['date']
        ser['id'] = file.name[5:8]
        ser['t'] = data[0]
        ser['A'] = data[1]
        ser['B'] = data[2]
        ser['meta'] = str(meta)
        ser['temp'] = meta.get('temp')
        ser['lock freq'] = meta.get('lock freq')
        ser['pos'] = meta.get('pos')
        return ser

    def toK(x):
        return thermocouple(tc_type='K', x=x, input_unit='mV', output_unit='K')

    def convert(ser, dic, type):
        for key, val in dic.items():
            if ser.get(key) is None:
                continue
            if val != 0:
                ser[key] = ser[key][:-val]
            ser[key] = int(ser[key])
        return ser

    def parseTemp(x):
        if x[-1] == 'K':
            return float(x[:-1])
        return toK(- float(x[:-2]))

    def parsePos(x):
        return float(x[:-1]) / 10

    # LOAD THE DATA AND PREPARE IT

    df = pd.DataFrame([TRtoSeries(file) for file in files])

    # fill different na values
    df['temp'] = df['temp'].fillna('293K')
    df['pos'] =  df['pos'].fillna('6933A')

    # converters
    d = {
        'id' : 0,
        'PT tension' : 1,
        'integration time' : 2,
        'output slit' : 3,
        'lock freq' : 2
    }

    f = {
        'input slit' : 3,
    }

    df = df.apply(convert, dic =d, type = int, axis = 1)
    df = df.apply(convert, dic =f, type = float, axis = 1)
    df['temp'] = df['temp'].apply(parseTemp).astype(float)
    df['pos'] = df['pos'].apply(parsePos)

    # meta is not useful anymore
    df = df.drop(columns = 'meta')


    df
    return (
        TRtoSeries,
        convert,
        d,
        df,
        f,
        files,
        parsePos,
        parseTemp,
        thermocouple,
        toK,
    )


@app.cell
def _(mo):
    mo.md(r"""# Fitting the data""")
    return


@app.cell
def _(FIG_SIZE, df, model, pd, plt, savefig, tnot):
    _fig, _axs = plt.subplots(8, 5, figsize=FIG_SIZE * 4)
    _axs = _axs.flatten()

    result_df = pd.DataFrame()

    # try to fit each row of the dataframe
    for _i, _batch in df.iterrows():
        _id, _date, _temp, _pos = _batch[['id', 'date', 'temp', 'pos']]
        print('____________________________', _id, _date, _temp, _pos)
        _t = _batch['t'] - _batch['t'][0]
        _A = -_batch['A']
        _params = model.make_params()
        _params['A'].set(0.7, min=0)
        _params['tau'].set(0.007, min=0.002, max=0.012)
        _params['T'].set(0.02, min=0.01)
        _params['tnot'].set(0.01)
        _result = model.fit(_A, _params, t=_t)

        # each batch is fitted in a different manner
        if int(_id) in range(10, 20):
            _params['tnot'].set(0.024)
            _params['T'].set(0.014)
            _params['tau'].set(0.01)

        if int(_id) in range(15, 21):
            _params['tnot'].set(0)
            _params['T'].set(0.02)
            _params['tau'].set(0.0095)
            _params['A'].set(0.27)

        if int(_id) in range(21, 25):
            _params['tnot'].set(0.01)
            _params['T'].set(0.014)
            _params['tau'].set(0.0095)
            _params['A'].set(0.27)

        _result = model.fit(_A, _params, t=_t)

        # try different starting points
        if _result.chisqr > 1:
            _tnot = _result.values['tnot']
            _T = _result.values['T']
            _params['tnot'].set(_tnot + _T / 2)
            _result = model.fit(_A, _params, t=_t)

        if _result.chisqr > 1:
            _tnot = _result.values['tnot']
            _T = _result.values['T']
            _params['tnot'].set(tnot + _T / 4)
            _result = model.fit(_A, _params, t=_t)



        # now we do the fit another time but estimating the weights

        #_std = (_A - _result.best_fit).std()
        #print(_std)
        #_weights = 1 / _std

        #_result = model.fit(_A, _params, t=_t, weights=_weights)


        # display the results
        print(_result.fit_report(show_correl=False))
        _axs[_i].plot(_t, _A)
        _axs[_i].plot(_t, _result.best_fit)
        _axs[_i].set_title(f'{_id}-{_date}-{_temp:.1f}-{_pos}', y=0.95)


        _batch['res'] = _result
        exclude_cols = ['t', 'A', 'B']  
        _batch = _batch.drop(index = exclude_cols)
        result_df = pd.concat([result_df, _batch], axis=1)

    _fig.supxlabel(r'wavelenght $[nm]$')
    _fig.tight_layout()
    _fig.subplots_adjust(left = 0.05, bottom=0.05, hspace=0.67, wspace=0)
    _fig.supylabel(r'e.m.F $[V]$')
    savefig(_fig, 'tau_batch' ,fig_size = FIG_SIZE * 4)
    plt.show()
    result_df = result_df.T
    result_df
    return exclude_cols, result_df


@app.cell
def _(mo):
    mo.md(r"""# Tau analysis""")
    return


@app.cell
def _(mo, result_df):
    import altair as alt
    from IPython.display import display
    def get_val_std(x):
        for p in ['tau', 'T', 'tnot', 'A']:
            x[p + '_val'] = x['res'].params[p].value
            x[p + '_std'] = x['res'].params[p].stderr
        return x

    expanded_result_df = result_df.apply(get_val_std, axis =1).drop(columns = 'res')
    expanded_result_df['expected lock freq'] = expanded_result_df['T_val'] ** -1
    display(expanded_result_df)

    chart = mo.ui.altair_chart(alt.Chart(expanded_result_df).mark_point().encode(
        x = 'temp',
        y = 'tau_val',
        color = 'pos',
    ))
    return alt, chart, display, expanded_result_df, get_val_std


@app.cell
def _(
    FIG_SIZE,
    OUT_PATHS,
    display,
    expanded_result_df,
    pd,
    plt,
    result_df,
    savefig,
):
    import matplotlib.lines as mlines
    from lmfit.models import LinearModel

    def _():
        # display only the ones we want to keep
        _mask = (expanded_result_df['lock freq'] > 50) & (expanded_result_df['temp'] < 150)
        _mask2 = (expanded_result_df['pos'] < 693) & (expanded_result_df['temp'] < 100)
        _mask = _mask | _mask2
        _mask = [not m for m in _mask]
        #_expanded_result_df = expanded_result_df[_mask]
        _expanded_result_df = expanded_result_df
        _first_peak_ids = list(range(2, 6)) + list(range(12, 18)) + list(range(23, 28))
        _second_peak_ids = list(range(2, 6)) + list(range(6, 12)) + list(range(28, 39))
        _first_df = _expanded_result_df[result_df['id'].isin(_first_peak_ids)]
        _second_df = _expanded_result_df[result_df['id'].isin(_second_peak_ids)]
        _first_df['type'] = 'First Peak'
        _second_df['type'] = 'Second Peak'
        display(_first_df.groupby('temp').agg({'tau_val' : ['mean', 'std']}))
        display(_second_df.groupby('temp').agg({'tau_val' : ['mean', 'std']}))
        _combined = pd.concat([_first_df,_second_df])
        _combined.to_latex(OUT_PATHS['results_dir'] + '/tau_info.txt', index=False, escape=False, na_rep = 'na')

        # Example figure size (define FIG_SIZE if not set)
        # Using seaborn styles
        with plt.style.context(['seaborn-v0_8-colorblind', 'seaborn-v0_8-paper']):
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            model = LinearModel()
            for _m, _df in zip(['s', '^'], [_first_df, _second_df]):
                x = _df['temp']
                y = _df['tau_val']
                result = model.fit(y, x = x)
                print(result.fit_report())
                ax.scatter(
                    x, y,
                    marker=_m, alpha=0.6, edgecolors='black')
                ax.plot(x, result.best_fit, '--', label = "linear regression")
            # Labels and title
            ax.set_xlabel("temperature [K]")
            ax.set_ylabel(r"$\tau$ value")
            ax.tick_params(direction="in", length=6, width=1.2)

        square_marker = mlines.Line2D([], [], color='blue', marker='s', linestyle='None', markersize=8, label='Small peak on the left')
        triangle_marker = mlines.Line2D([], [], color='green', marker='^', linestyle='None', markersize=8, label='Tall peak on the right')
        ax.legend(handles=[square_marker, triangle_marker])
        savefig(fig, 'tau')
        return plt.show()
    _()
    return LinearModel, mlines


@app.cell
def _(chart, mo):
    mo.vstack([chart, mo.ui.table(chart.value)])
    return


@app.cell
def _(mo):
    mo.md(r"""Now we make a""")
    return


@app.cell
def _(plt, result_df):
    _mask = (result_df['lock freq'] > 50) & (result_df['temp'] < 150)
    _mask = [not m for m in _mask]
    _batch = result_df[_mask]
    _tau = _batch['res'].apply(lambda x: x.values['tau'])
    _std = _batch['res'].apply(lambda x: x.params['tau'].stderr)
    _T = _batch['temp']
    _pos = _batch['pos'] / 10
    _chopper = _batch['lock freq'] ** (-1) / 2

    def get_marker(pos):
        if pos == 693.9 or pos == 695.5:
            return 1
        return 0

    _markers = ['o', 's', '^']
    _colors = []

    for _i in _pos.unique():
        _mask = _pos == _i
        _marker = _markers[get_marker(_i)]
        plt.scatter(_T[_mask], _tau[_mask], alpha=0.3, marker=_marker, c='k', label = get_marker(_i))
    plt.legend()
    plt.show()
    return (get_marker,)


@app.cell
def _(get_val_std, merged, utils):
    from math import floor, log10

    def parse_val_std(x, p):
        val = x.params[p].value
        _std = x.params[p].stderr
        _std = float(utils.display_sigfig(_std, 2))
        order_of_magnitude = floor(log10(abs(_std)))
        val = val / pow(10, order_of_magnitude)
        val = round(val)
        val = val * pow(10, order_of_magnitude)
        return f'{val} ± {_std}'



    fir_res = merged.apply(get_val_std, axis=1)
    to_keep = ['id', 'tau_val', 'tau_std', 'T_val', 'T_std', 'tnot_val', 'tnot_std', 'A_val', 'A_std']
    fir_res = fir_res[to_keep]
    fir_res
    return fir_res, floor, log10, parse_val_std, to_keep


@app.cell
def _(clean_df, fir_res, to_keep):
    clean_df_1 = clean_df.merge(fir_res, on='id')
    clean_df_1['fitted Hz'] = clean_df_1['T_val'] ** (-1)
    order = ['type', 'id', 'date', 'temp', 'lock freq', 'fitted Hz', 'PT tension']
    clean_df_1[order + to_keep]
    return clean_df_1, order


@app.cell
def _(clean_df_1):
    clean_df_1['lock freq'] = clean_df_1['lock freq'].apply(lambda x: float(x[:-2]))
    _mask = (clean_df_1['lock freq'] - clean_df_1['fitted Hz']).abs() < 1
    clean_df_2 = clean_df_1[_mask]
    clean_df_2
    return (clean_df_2,)


@app.cell
def _(clean_df_2, display, plt):
    _ids = list(range(2, 6))
    _ids = _ids + list(range(12, 18))
    _ids = _ids + list(range(23, 28))
    first_peak_df = clean_df_2[clean_df_2['id'].astype(int).isin(_ids)]
    display(first_peak_df)
    _T = first_peak_df['temp']
    _tau = first_peak_df['tau_val']
    _std = first_peak_df['tau_std']
    plt.errorbar(_T, _tau, yerr=_std, linestyle='', marker='.', alpha=0.5)
    plt.show()
    return (first_peak_df,)


@app.cell
def _(clean_df_2, display, plt):
    second_peak_df = clean_df_2[clean_df_2['id'].astype(int).isin(_ids)]
    display(second_peak_df)
    _T = second_peak_df['temp']
    _tau = second_peak_df['tau_val']
    _std = second_peak_df['tau_std']
    plt.errorbar(_T, _tau, yerr=_std, linestyle='', marker='.', alpha=0.5)
    plt.show()
    return (second_peak_df,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
