import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib as mpl
import matplotlib.font_manager as fm

# 改style要在改font之前
# plt.style.use('seaborn')

fm.fontManager.addfont('TaipeiSansTCBeta-Regular.ttf')
mpl.rc('font', family='Taipei Sans TC Beta')

TIME_SPAN_LEN = 21
CAPACITY = np.array([200, 150, 100, 70, 30, 10])
IND_DEMAND_MUL_SET = (0.5, 1, 2)
STAY_MUL_SET = (1/TIME_SPAN_LEN, 1/10, 1/5, 1/2)
# CAPACITY_MUL_SET = [[1, 1, 1, 1, 1, 1]]
CAPACITY_MUL_SET = []
all_capacity = CAPACITY.sum()
for c_id, c in enumerate(CAPACITY):
    down_mul = 1 - (c / (all_capacity - c))
    mul = np.repeat(down_mul, len(CAPACITY))
    mul[c_id] = 2
    CAPACITY_MUL_SET.append(mul)
CAPACITY_MUL_SET = np.array(CAPACITY_MUL_SET)

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
for stay_mul in STAY_MUL_SET:
    for capacity_mul in CAPACITY_MUL_SET:
        y_tick.append(f"stay length: {stay_mul: .2f}, hot room: {np.argmax(capacity_mul)}")

for ind_demand_mul in IND_DEMAND_MUL_SET:
    x_tick.append(f"{ind_demand_mul}")


lack = pd.read_csv("scores.csv")
plot_heatmap(lack, x_tick, y_tick, "Individual demand", "Factor about orders from agents", "要滿足所有訂單還需要成長幾倍的產能", "lack.png")
obj = pd.read_csv("objs.csv")
plot_heatmap(obj, x_tick, y_tick, "Individual demand", "Factor about orders from agents", "Objective Value", "obj.png")
