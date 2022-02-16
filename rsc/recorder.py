import numpy as np
import pandas as pd

from data_reader import JSONDataReader
from gurobi_optimizer import solve

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

scenario = {}
scores = np.zeros((24, 3))
objs = np.zeros((24, 3))
a = 0
for stay_mul in STAY_MUL_SET:
    for capacity_mul in CAPACITY_MUL_SET:
        scenario["a"] = f"stay_{stay_mul}_twicecapacity_{np.argmax(capacity_mul)}"
        b = 0
        for ind_demand_mul in IND_DEMAND_MUL_SET:
            scenario["i"] = f"ind_demand_{ind_demand_mul}"
            score = []
            obj = []
            for instance_id in range(10):
                data_reader = JSONDataReader(scenario)
                acceptance, upgrade, order_room_quantity, order_stay, capacity, obj_val = solve(data_reader, instance_id)
                order_room_quantity = pd.DataFrame.from_dict(order_room_quantity, orient="index").to_numpy()
                order_stay = pd.DataFrame.from_dict(order_stay, orient="index").to_numpy()
                capacity = pd.DataFrame.from_dict(capacity, orient="index").to_numpy()
                tmp = order_stay.sum(axis=1) * (1 - acceptance)
                s = (tmp.reshape((-1, 1)) * order_room_quantity).sum() / (order_stay.shape[1] * capacity.sum())
                score.append(s)
                obj.append(obj_val)
            scores[a, b] = np.mean(score)
            objs[a, b] = np.mean(obj)
            b+= 1
        a += 1
pd.DataFrame(scores).to_csv('scores.csv', index=False)
pd.DataFrame(objs).to_csv('objs.csv', index=False)
