import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

fm.fontManager.addfont('TaipeiSansTCBeta-Regular.ttf')
mpl.rc('font', family='Taipei Sans TC Beta')

def plot_heatmap(df, x_tick, y_tick, x_label, y_label, title, file_name):
    fig, ax = plt.subplots(figsize=(8, 14))
    sns.heatmap(df, square=False, annot=True, cbar=False, fmt='.2f', cmap="Blues")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # ax.set_xticks(np.arange(len(x_tick)) + 0.5)
    # ax.set_yticks(np.arange(len(y_tick)) + 0.5)
    ax.xaxis.set_ticklabels(x_tick)
    ax.yaxis.set_ticklabels(y_tick, rotation=45, va="center")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(file_name)