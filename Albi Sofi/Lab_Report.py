import marimo

__generated_with = "0.11.26"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Matter Physics Lab
        This notebook contains the data analysis for the following wl (wave lenght) spectra:
         1. [Black Body](#black-body-radiation) at T = 960 C.
         2. [Sodium lamp](#sodium-lamp-spectrum); 
         3. [He-Ne laser](#sodium-lamp-spectrum);

        The spectra are used to characterize the experimental setup.
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    from matplotlib import rcParams
    import modules.utils as utils
    from importlib import reload
    from functools import partial
    from typing import Literal
    reload(utils)
    OUT_PATHS = {'figs_dir': './figures', 'results_dir': './results'}
    IN_PATHS = {'black_body_dir': './raw_data/Black_Body_Radiation', 'na_dir': './raw_data/Na_Spectrum', 'ruby_dir': './raw_data/Ruby_Spectrum', 'laser_dir': './raw_data/Laser_Spectrum'}
    FIG_SIZE = np.array([6.4, 4])
    SAVE_FIG = True
    FIG_FMT = 'pdf'
    TRANSPARENT_PNG = True
    savefig = partial(utils.save_fig, fig_dir=OUT_PATHS['figs_dir'], fig_fmt=FIG_FMT, fig_size=FIG_SIZE, save=SAVE_FIG, transparent_png=TRANSPARENT_PNG)
    utils.check_paths(IN_PATHS, OUT_PATHS)

    def fmtax(_ax, label: bool | Literal['x', 'y'] = False, loc='upper left'):
        if label == True:
            _ax.set_xlabel('wavelength [nm]')
            _ax.set_ylabel('intensity [a.u.]')
        elif label == 'x':
            _ax.set_xlabel('wavelength [nm]')
        elif label == 'y':
            _ax.set_ylabel('intensity [a.u.]')
        _ax.legend(loc=loc, frameon=False)
        _ax.tick_params(direction='in', which='both')
        _ax.minorticks_on()
    return (
        FIG_FMT,
        FIG_SIZE,
        IN_PATHS,
        Literal,
        OUT_PATHS,
        SAVE_FIG,
        TRANSPARENT_PNG,
        fmtax,
        inset_axes,
        mark_inset,
        mo,
        np,
        partial,
        pd,
        plt,
        rcParams,
        reload,
        savefig,
        utils,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Black Body Radiation
        This is the cleaned spectrum of the Black Body. In this context the spectrum is 'cleaned' because the stepper motor encoder reads values with an error and the error stacks across the wl sweep.
        We took 6 slices of the spectrum and then sticked them together to get the full spectrum with an error that is almost constant accross all the wls. The error is not certain but it's for sure less than an Angstrom.
        """
    )
    return


@app.cell
def _(OUT_PATHS, fmtax, pd, plt, savefig):
    from os import path
    from IPython.display import Latex, display
    BB_Exp_df = pd.read_csv(path.join(OUT_PATHS['results_dir'], 'black_body.csv'), dtype='float64', converters={'meta': str})
    BB_Exp_df['wl'] = BB_Exp_df['wl'] / 10
    BB_Exp_df[['mean', 'std']] = BB_Exp_df[['mean', 'std']] * 1000
    with plt.style.context(['seaborn-v0_8-colorblind', 'seaborn-v0_8-paper']):
        _fig, _ax = plt.subplots(figsize=[6, 3.375])
        _ax.errorbar(BB_Exp_df['wl'], BB_Exp_df['mean'], yerr=BB_Exp_df['std'], label='Black Body Radiation')
        fmtax(_ax, label='x')
        _ax.set_ylabel('e.m.F [mV]')
        plt.show()
        savefig(_fig, fig_name='black_body_combined')
    return BB_Exp_df, Latex, display, path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Actually this plot is an errorbar plot but the errors are not visible beacuse they are too small:""")
    return


@app.cell
def _(BB_Exp_df):
    print(BB_Exp_df['mean'].agg(['min', 'max']).apply(lambda x: f"{x:.2e}"))
    print(BB_Exp_df['std'].agg(['min', 'max']).apply(lambda x: f"{x:.2e}"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here we want to fit the theoretical BB spectrum on top of the experimental measurements. We suppose they are proportional to eachother and fit for the proportionality constant. Only on the far left the proportionality relation holds so we fit in that region, after that the responce function is different from one and can be used to model the experimental responce.""")
    return


@app.cell
def _(BB_Exp_df, plt, utils):
    from scipy.optimize import curve_fit

    # fitting for the proportionality coeff
    fit = lambda x, a :  a * x
    wl = BB_Exp_df['wl']
    mean = BB_Exp_df['mean']
    theo = utils.plank(wl * 1e-9, 1240)

    print(utils.hc)

    # correting for single photon counting
    # plt.plot(wl, theo, label='Theoretical')
    photon_energy  = utils.hc /  (wl * 1e-9) 

    plt.plot(wl, theo / theo.max(), label='Theoretical')
    theo = theo / photon_energy
    plt.plot(wl, theo / theo.max(), label='Theoretical corrected')
    plt.show()
    return curve_fit, fit, mean, photon_energy, theo, wl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Get the responce function of the system. The first part of the responce it's noisy so it's best to do a linear regression""")
    return


@app.cell
def _(BB_Exp_df, OUT_PATHS, fmtax, mean, np, path, plt, savefig, theo, wl):
    from scipy.stats import linregress
    responce = mean / theo
    std = BB_Exp_df['std'] / theo
    responce = responce / responce.max()
    with plt.style.context(['seaborn-v0_8-colorblind', 'seaborn-v0_8-paper']):
        _fig, _ax = plt.subplots()
        _ax.scatter(wl, responce, c='orange', marker='.', label='Responce Fuction')
        print('until', wl[35])
        res = linregress(wl[:36], responce[:36])
        print(res)
        responce[:36] = np.polyval(res[:2], wl[:36])
        _ax.plot(wl, responce, label='Linear Fit')
        fmtax(_ax, label='x', loc='upper right')
        _ax.set(ylabel='relative intensity [norm.]')
        plt.show()
        savefig(_fig, fig_name='responce_fit')
    np.savetxt(path.join(OUT_PATHS['results_dir'], 'responce.csv'), np.array([wl, responce]).T, delimiter=',')
    return linregress, res, responce, std


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Sodium Lamp Spectrum
        The monochromator can have a slight shift in the measured wl. For calibrating the shift we use the known spectral lines of the sodium atoms in a sodium vapor lamp.
        First of all we import the dataframe
        """
    )
    return


@app.cell
def _(IN_PATHS, pd, utils):
    Na_files = utils.getFiles(IN_PATHS['na_dir'])
    Na_df = pd.DataFrame([utils.file_to_series(file) for file in Na_files])

    # convert angstroms to nanometers
    Na_df['wl'] = Na_df['wl'] / 10
    Na_df['samples'] = Na_df['wl'].apply(lambda x: len(x))
    Na_df
    return Na_df, Na_files


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""and we do some plotting to show all the data we gathered""")
    return


@app.cell
def _(FIG_SIZE, Na_df, OUT_PATHS, display, fmtax, pd, plt, savefig):
    from matplotlib import ticker
    import ast

    def get_info(x):
        meta = ast.literal_eval(x['meta'])
        ser = pd.Series(meta)
        d = {
            'PT tension' : 1,
            'integration time' : 2,   
            'lock freq' : 2,
            'output slit' : 3,
            'input slit' : 3,
        }

        f = {

        }

        for key, val in d.items():
            if ser.get(key) is None:
                continue
            if val != 0:
                ser[key] = ser[key][:-val]
            ser[key] = int(ser[key])

        for key, val in f.items():
            if ser.get(key) is None:
                continue
            if val != 0:
                ser[key] = ser[key][:-val]
            ser[key] = float(ser[key])

        wl = x['wl']
        ser['start'] = f'{wl.min():.3f}'
        ser['stop'] = f'{wl.max():.3f}'
        ser['step'] = f'{wl[1] - wl[0]:.3f}'
        ser['date'] = '-'.join([ser['date'][:2], ser['date'][2:4]])
        ser['slits'] = f'{ser.get('input slit')}, {ser.get('output slit')}'
        ser['lock freq'] = f'{ser.get('lock freq')}'

        return ser

    info = Na_df.apply(get_info, axis=1)
    info['lock freq'] = info['lock freq'].fillna(307)
    _order = ['type', 'id', 'date', 'integration time', 'PT tension', 'slits', 'lock freq','start', 'stop', 'step']
    info = info[_order]
    info['points'] = Na_df['samples']

    # display and save it
    display(info)
    info.to_latex(OUT_PATHS['results_dir'] + '/sodium_info.txt', index=False, escape=False, na_rep = 'na')

    def foo():
        with plt.style.context(['seaborn-v0_8-colorblind', 'seaborn-v0_8-paper']):
            _fig, axs = plt.subplots(4, 5, figsize=FIG_SIZE * 1.8, sharey=True)
            groups = Na_df.groupby('meta')
            for i, [name, group] in enumerate(groups):
                _ax = axs.flatten()[i]
                mean_1 = group.get('mean').values[0]
                wl_1 = group.get('wl').values[0]
                mean_1 = mean_1 / mean_1.max()
                _mask = mean_1 > mean_1.min() + 0.004
                mean_1 = mean_1[_mask]
                wl_1 = wl_1[_mask]
                _ax.plot(wl_1, mean_1)
                date = group.get('date').values[0]
                the_id = group.get('id').values[0]
                _ax.set_title(f'{the_id}-{date}', y=0.05)
            for _ax in axs.flatten():
                fmtax(_ax, label=False)
                _ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune='both'))
            _fig.tight_layout()
            _fig.subplots_adjust(left=0.1, wspace=0, right=0.95, hspace=0.25, top=0.95, bottom=0.1)
            _fig.supxlabel(r'wavelenght $[nm]$')
            _fig.supylabel(r'intensity $[a.u.]$')
            savefig(_fig, fig_name='na_spectrum_overview', fig_size=FIG_SIZE * 1.8)
            plt.show()
    foo()
    return ast, foo, get_info, info, ticker


@app.cell
def _(FIG_SIZE, Na_df, fmtax, plt, savefig, ticker):
    with plt.style.context(['seaborn-v0_8-colorblind', 'seaborn-v0_8-paper']):
        _fig, axs = plt.subplots(3, 5, figsize=FIG_SIZE * 1.2, sharey=True)
        _tokeep = Na_df[Na_df['id'].astype(int).isin(range(10))]
        _mask = (_tokeep['id'].astype(int).isin(range(5))) & (_tokeep['date'] == '29102024')
        _mask = [not i for i in _mask]
        print(_mask)
        _tokeep = _tokeep[_mask]
        groups = _tokeep.groupby('meta')
        for i, [name, group] in enumerate(groups):
            _ax = axs.flatten()[i]
            mean_1 = group.get('mean').values[0]
            wl_1 = group.get('wl').values[0]
            mean_1 = mean_1 / mean_1.max()
            _mask = mean_1 > mean_1.min() + 0.004
            mean_1 = mean_1[_mask]
            wl_1 = wl_1[_mask]
            _ax.plot(wl_1, mean_1)
            date = group.get('date').values[0]
            the_id = group.get('id').values[0]
            _ax.set_title(f'{the_id}-{date}', y=1)
        for _ax in axs.flatten():
            fmtax(_ax, label=False)
            _ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune='both'))
        _fig.tight_layout()
        _fig.subplots_adjust(left=0.1, wspace=0, right=0.95, hspace=0.25, top=0.95, bottom=0.1)
        _fig.supxlabel(r'wavelenght $[nm]$')
        _fig.supylabel(r'intensity $[a.u.]$')
        savefig(_fig, fig_name='na_spectrum_overview', fig_size=FIG_SIZE * 1.8)
        plt.show()
    return axs, date, group, groups, i, mean_1, name, the_id, wl_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We introduced an iris to better focus the light in the monochromator.
        Its effect is displayed below.
        """
    )
    return


@app.cell
def _(FIG_SIZE, Na_df, fmtax, plt, savefig):
    a = Na_df[(Na_df['id'] == '008') & (Na_df['date'] == '29102024')].copy()
    b = Na_df[(Na_df['id'] == '009') & (Na_df['date'] == '29102024')].copy()
    with plt.style.context(['seaborn-v0_8-paper']):
        _fig, _ax = plt.subplots(figsize=FIG_SIZE)
        for item, fmt in zip([a, b], ['-', '-']):
            wl_2 = item['wl'].values[0]
            mean_2 = item['mean'].values[0] / item['mean'].values[0].max()
            _ax.plot(wl_2, mean_2, fmt, label=f'Na Spectrum {item['id'].values[0]}')
        _ax.legend(loc='upper right', frameon=False)
        fmtax(_ax, label=True)
        savefig(_fig, fig_name='na_iris_effect', fig_size=FIG_SIZE)
        plt.show()
    return a, b, fmt, item, mean_2, wl_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Fitting the data at 800 nm
        There should be a noticeable difference between the raw data and the correction of the responce function
        """
    )
    return


@app.cell
def _(OUT_PATHS, np, path, plt):
    from scipy.interpolate import interp1d
    responce_1 = np.loadtxt(path.join(OUT_PATHS['results_dir'], 'responce.csv'), delimiter=',').T
    plt.plot(responce_1[0], responce_1[1])
    plt.show()
    return interp1d, responce_1


@app.cell
def _(fmtax, interp1d, np, plt, savefig):
    from lmfit import Minimizer, create_params, report_fit
    from lmfit.lineshapes import gaussian, lorentzian, voigt, linear
    from lmfit.models import VoigtModel, LinearModel, ConstantModel

    def fit_Na(_sample, model, _params):
        wl = _sample['wl'].values[0]
        mean = _sample['mean'].values[0]
        std = _sample['std'].values[0]
        weights = np.maximum(np.abs(std), 1e-06 / np.sqrt(3))
        weights = 1 / weights
        _result = model.fit(mean, _params, weights=weights, x=wl)
        return _result

    def plot_Na(_sample, _result, fig_name):
        kw = {'marker': '.', 'capsize': 2, 'linestyle': 'none', 'alpha': 0.7}
        wl = _sample['wl'].values[0]
        with plt.style.context(['seaborn-v0_8-colorblind', 'seaborn-v0_8-paper']):
            _fig = _result.plot(data_kws=kw, title=' ')
            _fig.tight_layout()
            _fig.subplots_adjust(hspace=0.08)
            components = _result.eval_components(params=_result.params, x=wl)
            axs = _fig.get_axes()
            axs[0].tick_params(direction='in', which='both')
            ax_peak = _fig.add_subplot(313, sharex=axs[0], sharey=axs[1])
            ax_peak.set_ylabel('peaks [a.u.]')
            for key in components.keys():
                if key == 'bkg_':
                    continue
                label = _result.params[f'{key}center'].value
                label = f'{label:.2f} nm'
                ax_peak.plot(wl, components[key], label=label)
                ax_peak.fill_between(wl, components[key], 0, alpha=0.2)
            pos = ax_peak.get_position()
            delta = pos.height + 0.06
            ax_peak.set_position([pos.x0, pos.y0 - delta, pos.width, pos.height])
            ax_peak.tick_params(direction='in', which='both')
            ax_peak.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax_peak.legend(loc='upper right', frameon=False)
            fmtax(axs[1], loc='upper right', label='y')
            axs[1].set_xlabel(None)
            ax_peak.set_xlabel('wavelength [nm]')
            plt.show()
            savefig(_fig, fig_name=fig_name)

    def correct_sample(_sample, responce):
        wl = _sample['wl'].values[0]
        mean = np.array(_sample['mean'].values[0])
        std = np.array(_sample['std'].values[0])
        interp = interp1d(responce[0], responce[1], kind='linear')(wl)
        mean = mean * interp
        std = std * interp
        _sample['mean'] = [mean]
        _sample['std'] = [std]
        return _sample
    return (
        ConstantModel,
        LinearModel,
        Minimizer,
        VoigtModel,
        correct_sample,
        create_params,
        fit_Na,
        gaussian,
        linear,
        lorentzian,
        plot_Na,
        report_fit,
        voigt,
    )


@app.cell
def _(
    LinearModel,
    Na_df,
    VoigtModel,
    ast,
    correct_sample,
    display,
    fit_Na,
    plot_Na,
    responce_1,
):
    _mask = (Na_df['date'] == '17102024') & (Na_df['id'] == '005')
    _sample = Na_df[_mask]
    _sample = correct_sample(_sample, responce_1)
    display(ast.literal_eval(_sample['meta'].values[0]))
    model = VoigtModel(prefix='one_') + VoigtModel(prefix='two_') + LinearModel(prefix='bkg_')
    _params = model.make_params()
    _params['one_center'].set(589.453201, min=589, max=590.5)
    _params['one_sigma'].set(0.06926168)
    _params['one_gamma'].set(0.02878824, vary=True)
    _params['one_amplitude'].set(0.00020098, min=1e-05)
    _params['two_center'].set(590.020911, min=589, max=590.5)
    _params['two_sigma'].set(0.08952652)
    _params['two_gamma'].set(-0.00139832, vary=True)
    _params['two_amplitude'].set(0.00012348, min=1e-05)
    _params['bkg_slope'].set(6.132099e-08, vary=True)
    _params['bkg_intercept'].set(-3.62328e-05, vary=True)
    _result = fit_Na(_sample, model, _params)
    print(_result.fit_report())
    plot_Na(_sample, _result, 'Na_17102024_005')
    return (model,)


@app.cell
def _(Na_df, correct_sample, fit_Na, model, plot_Na, responce_1):
    _mask = (Na_df['date'] == '29102024') & (Na_df['id'] == '009')
    _sample = Na_df[_mask]
    _sample = correct_sample(_sample, responce_1)
    _params = model.make_params()
    _params['one_center'].set(818.5)
    _params['one_sigma'].set(0.009)
    _params['one_gamma'].set(0.011, vary=True)
    _params['one_amplitude'].set(3.67e-05)
    _params['two_center'].set(819.6)
    _params['two_sigma'].set(0.009)
    _params['two_gamma'].set(0.0074, vary=True)
    _params['two_amplitude'].set(3.67e-05)
    _params['bkg_slope'].set(0, vary=True)
    _params['bkg_intercept'].set(0, vary=True)
    _result = fit_Na(_sample, model, _params)
    print(_result.fit_report())
    plot_Na(_sample, _result, 'Na_29102024_009')
    return


@app.cell
def _(mo):
    mo.md(r"""# Laser Spectrum""")
    return


@app.cell
def _(IN_PATHS, pd, utils):
    lsr_files = utils.getFiles(IN_PATHS['laser_dir'])
    lsr_df = pd.DataFrame([utils.file_to_series(file) for file in lsr_files])

    # filter ids
    lsr_df['id'] = lsr_df['id'].astype(int)
    _to_keep = list(range(4))
    lsr_df = lsr_df[lsr_df['id'].isin(_to_keep)]

    # convert angstroms to nanometers
    lsr_df['wl'] = lsr_df['wl'] / 10
    lsr_df['start'] = lsr_df['wl'].apply(min)
    lsr_df['stop'] = lsr_df['wl'].apply(max)
    lsr_df = lsr_df.drop(columns = 'interval')
    lsr_df['samples'] = lsr_df['wl'].apply(lambda x: len(x))
    lsr_df
    return lsr_df, lsr_files


@app.cell
def _(
    FIG_SIZE,
    OUT_PATHS,
    display,
    fmtax,
    get_info,
    lsr_df,
    plt,
    savefig,
    ticker,
):
    _info = lsr_df.apply(get_info, axis=1)
    _info['lock freq'] = _info['lock freq'].fillna(307)
    _order = ['type', 'id', 'date', 'integration time', 'PT tension', 'slits', 'lock freq','start', 'stop', 'step']
    _info = _info[_order]
    _info['points'] = lsr_df['samples']

    # display and save it
    display(_info)
    _info.to_latex(OUT_PATHS['results_dir'] + '/lsr_info.txt', index=False, escape=False, na_rep = 'na')

    def fooo():
        with plt.style.context(['seaborn-v0_8-colorblind', 'seaborn-v0_8-paper']):
            _fig, axs = plt.subplots(1, 5, figsize=FIG_SIZE * [4,1], sharey=True)
            groups = lsr_df.groupby('meta')
            for i, [name, group] in enumerate(groups):
                _ax = axs.flatten()[i]
                mean_1 = group.get('mean').values[0]
                wl_1 = group.get('wl').values[0]
                mean_1 = mean_1 / mean_1.max()
                _mask = mean_1 > mean_1.min() + 0.004
                mean_1 = mean_1[_mask]
                wl_1 = wl_1[_mask]
                _ax.plot(wl_1, mean_1)
                date = group.get('date').values[0]
                the_id = group.get('id').values[0]
                _ax.set_title(f'{the_id}-{date}', y=0.05)
            for _ax in axs.flatten():
                fmtax(_ax, label=False)
                _ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune='both'))
            _fig.tight_layout()
            _fig.subplots_adjust(left=0.05, wspace=0, right=0.95, hspace=0.25, top=0.95, bottom=0.15)
            _fig.supxlabel(r'wavelenght $[nm]$')
            _fig.supylabel(r'intensity $[a.u.]$')
            savefig(_fig, fig_name='lsr_spectrum_overview', fig_size=FIG_SIZE * [2.5,1])
            plt.show()
    fooo()
    return (fooo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
