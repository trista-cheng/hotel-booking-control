import json

import numpy as np
import pandas as pd

from plotter import plot_heatmap

# 改style要在改font之前
# plt.style.use('seaborn')

with open("scenarios.json") as f:
    scenarios = json.load(f)

y_tick = []
x_tick = []
for agent_factor in scenarios["agent"]:
    segs = agent_factor.split('_')
    y_tick.append(f"stay length: {segs[2]}, hot room: {segs[-1]}")

for ind_factor in scenarios["individual"]:
    x_tick.append(ind_factor.split("_")[-1])

lack = pd.read_csv("lacks.csv", index_col=0)
plot_heatmap(lack, x_tick, y_tick, "Individual demand", "Factor about orders from agents", "被拒絕的訂單所需房間數對產能的比值", "lack.png")
obj = pd.read_csv("objs.csv", index_col=0)
plot_heatmap(obj, x_tick, y_tick, "Individual demand", "Factor about orders from agents", "Objective Value", "obj.png")
