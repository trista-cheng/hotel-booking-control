import json
import numpy as np
import pandas as pd
import xarray as xr

from os.path import join
from pathlib import Path
from time import perf_counter
from data_reader import JSONDataReader
from gurobi_optimizer import solve

with open("scenarios.json") as f:
    scenarios = json.load(f)

SOLUTION_DIR = "solution"

lacks = []
objs = []
times = []
for agent_factor in scenarios["agent"]:
    scenario = {}
    scenario["agent"] = agent_factor
    lacks.append([])
    objs.append([])
    times.append([])
    for ind_factor in scenarios["individual"]:
        scenario["individual"] = ind_factor
        lack = []
        obj = []
        cal_time = []
        for instance_id in range(2):
            start_time = perf_counter()
            data_reader = JSONDataReader(scenario)
            # acceptance: 1 x order
            # upgrade: order x room x room
            (acceptance, upgrade, order_room_quantity, order_stay, 
             capacity, obj_val) = solve(data_reader, instance_id)
            output_folder = join(SOLUTION_DIR, agent_factor + ind_factor)
            Path(output_folder).mkdir(exist_ok=True, parents=True)
            pd.DataFrame(acceptance).to_csv(
                join(output_folder, f"{instance_id}_acceptance.csv"), 
                index=False
            )
            coords={
                "order": np.arange(acceptance.shape[0]) + 1,
                "room": np.arange(upgrade.shape[1]) + 1,
                "up_room": np.arange(upgrade.shape[1]) + 1,
            }
            # for up_type in range(upgrade.shape[1]):
            #     coords[f"up_to_{up_type + 1}"] = upgrade[:, :, up_type].flatten()
            upgrade_data = xr.DataArray(
                upgrade,
                dims=("order", "room", "up_room"),
                coords=coords,
            )
            upgrade_data.to_netcdf(join(output_folder, f"{instance_id}_upgrade.nc"))
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

pd.DataFrame(lacks, index=scenarios["agent"], columns=scenarios["individual"]).to_csv('lacks.csv', index=True)
pd.DataFrame(objs, index=scenarios["agent"], columns=scenarios["individual"]).to_csv('objs.csv', index=True)
pd.DataFrame(times, index=scenarios["agent"], columns=scenarios["individual"]).to_csv("time_log.csv", index=True)
