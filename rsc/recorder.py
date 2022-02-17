import json
import numpy as np
import pandas as pd

from time import perf_counter
from data_reader import JSONDataReader
from gurobi_optimizer import solve

with open("scenarios.json") as f:
    scenarios = json.load(f)

lacks = []
objs = []
times = []
for agent_factor in scenarios["agent"][:1]:
    scenario = {}
    scenario["agent"] = agent_factor
    lacks.append([])
    objs.append([])
    times.append([])
    for ind_factor in scenarios["individual"][:1]:
        scenario["individual"] = ind_factor
        lack = []
        obj = []
        cal_time = []
        for instance_id in range(1):
            start_time = perf_counter()
            data_reader = JSONDataReader(scenario)
            acceptance, upgrade, order_room_quantity, order_stay, capacity, obj_val = solve(data_reader, instance_id)
            order_room_quantity = pd.DataFrame.from_dict(order_room_quantity, orient="index").to_numpy()
            order_stay = pd.DataFrame.from_dict(order_stay, orient="index").to_numpy()
            capacity = pd.DataFrame.from_dict(capacity, orient="index").to_numpy()
            stay_sum = order_stay.sum(axis=1) * (1 - acceptance)
            lack_value = (stay_sum.reshape((-1, 1)) * order_room_quantity).sum() / (order_stay.shape[1] * capacity.sum())
            lack.append(lack_value)
            obj.append(obj_val)
            cal_time.append(perf_counter() - start_time)
        lacks[-1].append(np.mean(lack))
        objs[-1].append(np.mean(obj))
        times[-1].append(np.mean(cal_time))
# TODO data should with well-defined index and columns
pd.DataFrame(lacks).to_csv('lacks.csv', index=False)
pd.DataFrame(objs).to_csv('objs.csv', index=False)
pd.DataFrame(times).to_csv("time_log.csv", index=False)
