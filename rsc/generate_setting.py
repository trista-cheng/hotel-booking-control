from pathlib import Path
from os.path import join

import configparser
import copy
import numpy as np

def convert_name(variable, digit=2):
    if type(variable) == float:
        variable = np.round(variable, digit)
    string = str(variable)
    string = string.replace('.', '')
    return string


base_setting = {
    # generate data setting
    # "output_root": "mid_data",
    "replicate_num": 10,
    "batch_size": 10,
    "room_request_ratio_threshold": 2,
    # data outline and ratio 
    "time_span_len": 30,
    "num_room_type": 12,
    "padding": 0.4,
    "num_room_multiplier": 0.1,
    "price_multiplier": 0.8,
    "upgrade_fee_gap_multiplier": 0.3,
    # standard level data
    "capacity": np.array([
        200, 180, 170, 150, 130, 100, 90, 80, 75, 50, 
        30, 20
    ]),
    "individual_price": np.array([
        900, 1000, 1200, 1300, 1500, 1800, 2100, 2500, 2700, 2800, 
        4000, 5000,
    ]),
    "individual_pop_size": np.array([
        100, 90, 85, 75, 65, 50, 45, 40, 37, 25, 
        25, 25
    ]),
    "weekend_rate": np.array([
        0.4, 0.35, 0.3, 0.35, 0.3, 0.25, 0.3, 0.2, 0.1, 0.25, 
        0.15, 0.2
    ]),
    "week_rate": np.array([
        0.3, 0.3, 0.2, 0.3, 0.2, 0.2, 0.2, 0.1, 0.05, 0.15, 
        0.1, 0.15,
    ]),
}

# factor range
IND_DEMAND_MUL_SET = (0.5, 1, 2)
STAY_MUL_SET = (1/4, 1/5, 1/7, 1/9)
ROOM_RATE_SET = np.array([
    np.array([
        0.05, 0.1, 0.2, 0.3, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 
        0.1, 0.05
    ]),
    np.array([
        0.05, 0.05, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 
        0.4, 0.5
    ]),
    np.array([
        0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1, 
        0.05, 0.05
    ]),
])


config = configparser.ConfigParser(
    interpolation=configparser.ExtendedInterpolation()
)

folder = "scenarios"
config['META'] = {
    'root': folder,
    'factor': "stay duration, room rate, individual demand"
}
config['FACTOR'] = {
    "stay_duration": '\n' + '\t\n'.join([str(v) for v in STAY_MUL_SET]),
    "room_rate": '\n' + '\t\n'.join([str(v) for v in ROOM_RATE_SET]),
    "individual_demand": '\n' + "\t\n".join(
        [str(v) for v in IND_DEMAND_MUL_SET]
    ),
}
file_path = []
Path(folder).mkdir(parents=True, exist_ok=True)
for ind_demand_mul in IND_DEMAND_MUL_SET:
    for stay_mul in STAY_MUL_SET:
        for room_rate in ROOM_RATE_SET:
            setting = copy.deepcopy(base_setting)
            setting['ind_demand_mul'] = ind_demand_mul
            setting['stay_mul'] = stay_mul
            setting['room_rate'] = room_rate
            setting['display_name'] = (
                f"Stay duration: {stay_mul:.2f}| "
                f"Highest room rate: {room_rate.argmax()}| "
                f"Individual demand: {ind_demand_mul:.2f}"
            )
            file_name = (
                f"stay_{convert_name(stay_mul)}__room_{room_rate.argmax()}"
                f"__ind_{convert_name(ind_demand_mul)}.npy"
            )
            setting['file_path'] = join(folder, file_name)
            with open(join(folder, file_name), 'wb') as f: 
                np.save(f, setting)
            file_path.append(f"${{META:root}}/{file_name}")
config['SCENARIO'] = {"file_path": '\t\n'.join(file_path)}
with open('scenario.ini', 'w') as f:
    config.write(f)
