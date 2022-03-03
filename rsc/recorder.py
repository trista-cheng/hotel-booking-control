import json
import numpy as np
import pandas as pd
import xarray as xr

from os.path import join
from pathlib import Path
from time import perf_counter
from data_reader import CSVDataReader, JSONDataReader
from gurobi_optimizer import solve
from metric import get_reject_room_ratio

with open("scenarios.json") as f:
    scenarios = json.load(f)

SOLUTION_DIR = "solution"
INSTANCE_NUM = 10

lacks = []
objs = []
times = []
for agent_factor in scenarios["agent"]:
    scenario = {}
    scenario["agent"] = agent_factor
    for ind_factor in scenarios["individual"]:
        scenario["individual"] = ind_factor
        lack = []
        obj = []
        time = []
        for instance_id in range(INSTANCE_NUM):
            start_time = perf_counter()
            data_reader = JSONDataReader(scenario)
            # acceptance: 1 x order
            # upgrade: order x room x room
            acceptance, upgrade, obj_val = solve(data_reader, instance_id)
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
            time.append(perf_counter() - start_time)
        lacks.append(np.mean(lack))
        objs.append(np.mean(obj))
        times.append(np.mean(time))

data = xr.DataArray(
    pmf,
    dims=("room", "time", "outcome"),
    coords={
        "room": np.arange(self.num_room_type) + 1,
        "time": np.arange(self.time_span_len) + 1,
        "outcome": np.arange(pmf.shape[2]) + 1,
        "quantity": ("outcome", np.arange(pmf.shape[2]))
    }
)
pmf_dict = data.to_dataframe(name="prob")
pd.DataFrame(lacks, index=scenarios["agent"], columns=scenarios["individual"]).to_csv('lacks.csv', index=True)
pd.DataFrame(objs, index=scenarios["agent"], columns=scenarios["individual"]).to_csv('objs.csv', index=True)
pd.DataFrame(times, index=scenarios["agent"], columns=scenarios["individual"]).to_csv("time_log.csv", index=True)
