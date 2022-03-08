import json
import numpy as np
import pandas as pd
import xarray as xr
import logging

from os.path import join
from pathlib import Path
from time import perf_counter

from data_reader import CSVDataReader, JSONDataReader
from gurobi_optimizer import solve
from metric import get_reject_room_ratio
from validator import Validator

with open("scenarios.json") as f:
    scenarios = json.load(f)

logging.basicConfig(filename='log.log', 
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.WARNING, datefmt='%Y-%m-%d %H:%M:%S')
logging.warning("Start!")

SOLUTION_DIR = "solution"
INSTANCE_NUM = 2

lacks = []
objs = []
times = []
index = []
for agent_factor in scenarios["agent"]:
    scenario = {}
    scenario["agent"] = agent_factor
    for ind_factor in scenarios["individual"]:
        scenario["individual"] = ind_factor
        index.append((agent_factor.split('_')[2], agent_factor.split('_')[-1], ind_factor.split('_')[-1]))
        lack = []
        obj = []
        cal_time = []
        for instance_id in range(INSTANCE_NUM):
            start_time = perf_counter()
            data_reader = JSONDataReader(scenario)
            # acceptance: 1 x order
            # upgrade: order x room x room
            acceptance, upgrade, obj_val, sale = solve(data_reader, instance_id)

            test = Validator(scenario, instance_id, acceptance, upgrade, sale)
            test.validate_shape()
            test.validate_capacity_obj(obj_val)

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
            lack_value = get_reject_room_ratio(scenario, instance_id, 
                                               acceptance)
            lack.append(lack_value)
            obj.append(obj_val)
            cal_time.append(perf_counter() - start_time)
            logging.warning(f"{agent_factor}, {ind_factor}, {instance_id}")
        lacks.append(np.mean(lack))
        objs.append(np.mean(obj))
        times.append(np.mean(cal_time))

result = np.hstack(
    [np.array(objs).reshape((-1, 1)), np.array(lacks).reshape((-1, 1)), np.array(times).reshape((-1, 1))]
)
index = pd.MultiIndex.from_tuples(index, names=["stay mul", "high request room", "ind demand mul"])
pd.DataFrame(result, index=index, columns=["obj", "reject capacity ratio", "time"]).to_csv("result.csv")
logging.warning("End!")