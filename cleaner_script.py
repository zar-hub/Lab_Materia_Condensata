'''
A simple python app to clean experimental data.
Features:
- renaming files
- selecting partial data
'''
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import SpanSelector, TextBox, Button
import sys
import modules.utils as utils
import pandas as pd
import os

# globals
cwd = os.getcwd()
col_names = ['wl', 'mean', '1', '2', '3']
save_path = os.path.join(cwd, 'Cleaned_Data')

def app(files : list | os.DirEntry):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))
    
    class Index:
        ind = 0
        save_name = ''

        def load(self):
            # modify this to be more generic
            df = pd.read_csv(files[self.ind].path, 
                            sep=r'    ',
                            names=col_names,
                            engine='python',
                            dtype=np.float32)
            line = ax1.lines[0]
            x, y = df['wl'].to_numpy(),df['mean'].to_numpy()
            line.set_data(x,y)   
            ax1.set_xlim(x.min(), x.max())
            ax1.set_ylim(y.min(), y.max())
            # reset ax2
            ax2.lines[0].set_data([],[])
            fig.canvas.draw_idle()
            # update savename
            self.save_name = files[self.ind].name

        def next(self, event=''):
            # save current file
            line = ax2.lines[0]
            x, y = line.get_data()
            print(x,y)
            df = pd.DataFrame(np.asarray([x,y]).T, columns=['wl', 'mean'], dtype=np.float32)
            df['meta'] = files[self.ind].name
            df.to_csv(os.path.join(save_path, self.save_name))
            
            # load the next
            self.ind += 1
            i = self.ind % len(files)
            self.load()

        def prev(self, evnet=''):
            self.ind -= 1
            i = self.ind % len(files)
            self.load()
            
    # always convert the files as an array
    if type(files) is os.DirEntry:
        files = [files]
        
    callback = Index()
    
    input_ax = fig.add_axes([0.15, .95, .8, .05])
    button_ax = fig.add_axes([.4, 0, .1, .05])
    button = Button(button_ax, 'save')

    ax1.set_title('Slice the data you want to keep')
    
    line1, = ax1.plot([], [])
    line2, = ax2.plot([], [])

    # hooks
    def onselect(xmin, xmax):
        x = line1.get_xdata()
        y = line1.get_ydata()
        indmin, indmax = np.searchsorted(x, (xmin, xmax))
        indmax = min(len(x) - 1, indmax)

        region_x = x[indmin:indmax]
        region_y = y[indmin:indmax]

        if len(region_x) >= 2:
            line2.set_data(region_x, region_y)
            ax2.set_xlim(region_x[0], region_x[-1])
            ax2.set_ylim(region_y.min(), region_y.max())
            fig.canvas.draw_idle()

    def set_filename(fname):
        if fname == '':
            return
        input_box.set_val(fname)
        print(f'rename file: {fname}')
        callback.save_name = fname
        
    # WIDGETS
    span = SpanSelector(
        ax1,
        onselect,
        "horizontal",
        useblit=True,
        props=dict(alpha=0.5, facecolor="tab:blue"),
        interactive=True,
        drag_from_anywhere=True
    )
    # Set useblit=True on most backends for enhanced performance.
    input_box = TextBox(
        input_ax,
        'rename file',
        files[0].name,
        textalignment='left'
    )
    
    # interactivity
    input_box.on_submit(set_filename)
    button.on_clicked(callback.next)
    button.on_clicked(lambda e : input_box.set_val(files[callback.ind].name))
    
    # manually load the first data
    callback.load()
    plt.show()
    
def print_header(script_name, folder_path):
    header = f"""
    ============================================
    |          Experimental Data Cleaner        |
    ============================================
    Script: {script_name}
    Folder: {folder_path}
    ============================================
    """
    print(header)
    
def main(path : str, quiet = False):
    print_header(sys.argv[0], path)
    files = utils.getFiles(path)
    for f in files:
        print(f)
        
    if utils.query_yes_no('\nproceed?') is False:
        return
    
    # start the app
    app(files)

if __name__ == '__main__':
    from argh import dispatch_command
    dispatch_command(main)
else:
    app()