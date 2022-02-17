import json
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib as mpl
import matplotlib.font_manager as fm

# 改style要在改font之前
# plt.style.use('seaborn')

with open("scenarios.json") as f:
    scenarios = json.load(f)

fm.fontManager.addfont('TaipeiSansTCBeta-Regular.ttf')
mpl.rc('font', family='Taipei Sans TC Beta')

def plot_heatmap(df, x_tick, y_tick, x_label, y_label, title, file_name):
    fig, ax = plt.subplots(figsize=(8, 14))
    sns.heatmap(df, square=False, annot=True, cbar=False, fmt='.2f', cmap="Blues")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.xaxis.set_ticklabels(x_tick)
    ax.yaxis.set_ticklabels(y_tick, rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(file_name)

y_tick = []
x_tick = []
for agent_factor in scenarios["agent"]:
    segs = agent_factor.split('_')
    y_tick.append(f"stay length: {segs[2]}, hot room: {segs[-1]}")

for ind_factor in scenarios["individual"]:
    x_tick.append(ind_factor.split("_")[-1])

lack = pd.read_csv("scores.csv")
plot_heatmap(lack, x_tick, y_tick, "Individual demand", "Factor about orders from agents", "要滿足所有訂單還需要成長幾倍的產能", "lack.png")
obj = pd.read_csv("objs.csv")
plot_heatmap(obj, x_tick, y_tick, "Individual demand", "Factor about orders from agents", "Objective Value", "obj.png")
