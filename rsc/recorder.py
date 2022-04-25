import json
import numpy as np
import pandas as pd
import xarray as xr
import logging
import configparser

from os.path import join
from pathlib import Path
from time import perf_counter

from data_reader import JSONDataReader
from gurobi_optimizer import solve
# from metric import get_reject_room_ratio


logging.basicConfig(filename='log.log',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.WARNING, datefmt='%Y-%m-%d %H:%M:%S')
logging.warning("Start!")


UPGRADE_RULE = "up"
SOLUTION_DIR = f"solutions/{UPGRADE_RULE}"
INSTANCE_NUM = 3
DATA_ROOT = "data"

# TODO consider to put index information in scenarios.ini
def get_index(scenario):
    room_rate = np.array(scenario['room_rate'][1:-1].split(), dtype=float)
    hot_room = room_rate.argmax()
    return [float(scenario['stay_mul']), hot_room, float(scenario['ind_demand_mul'])]

scenarios = configparser.ConfigParser()
scenarios.read('scenarios.ini')

# lacks = []
objs = []
times = []
index = []
for scenario_name in scenarios.sections():
    scenario = scenarios[scenario_name]
    index.append(get_index(scenario))

    # lack = []
    obj = []
    cal_time = []
    for instance_id in range(INSTANCE_NUM):
        start_time = perf_counter()
        data_reader = JSONDataReader(scenario, data_root=DATA_ROOT)
        # acceptance: 1 x order
        # upgrade: order x room x room
        acceptance, upgrade, obj_val, sale = solve(data_reader, instance_id, UPGRADE_RULE)

        # test = Validator(scenario, instance_id, acceptance, upgrade, sale)
        # try:
        #     test.validate_shape(rule=UPGRADE_RULE)
        #     test.validate_capacity_obj(obj_val)
        # except:
        #     continue

        output_folder = join(SOLUTION_DIR, scenario_name)
        Path(output_folder).mkdir(exist_ok=True, parents=True)
        sale.to_csv(join(output_folder, f"{instance_id}_sale.csv"))
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
        # lack_value = get_reject_room_ratio(scenario, instance_id,
        #                                     acceptance, data_root=DATA_ROOT)
        # lack.append(lack_value)
        obj.append(obj_val)
        cal_time.append(perf_counter() - start_time)
        logging.warning(f"{scenario_name}, {instance_id}")
    pd.DataFrame({"time": cal_time, "obj": obj}).to_csv(
        join(output_folder, "time_obj.csv"), index=False
    )
    # lacks.append(np.mean(lack))
    objs.append(np.mean(obj))
    times.append(np.mean(cal_time))

result = np.hstack([
    np.array(objs).reshape((-1, 1)),
    # np.array(lacks).reshape((-1, 1)),
    np.array(times).reshape((-1, 1))
])
index = pd.MultiIndex.from_tuples(
    index,
    names=["stay mul", "high request room", "ind demand mul"]
)
pd.DataFrame(
    result,
    index=index,
    columns=["obj", "time"]  # "reject capacity ratio",
).to_csv(f"avg_result_{UPGRADE_RULE}.csv")
logging.warning("End!")