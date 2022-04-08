from pathlib import Path
from os.path import join, exists

import configparser
import copy
import numpy as np

def convert_name(variable, digit=2):
    if type(variable) == float:
        variable = np.round(variable, digit)
    string = str(variable)
    string = string.replace('.', '')
    return string


basic_setting = {
    # hotel basic info
    "time_span_len": 14,
    "num_room_type": 6,
    # standard level data
    "capacity": np.array([
        200, 180, 150, 100, 80, 50,
    ]),
    "individual_price": np.array([
        900, 1300, 1500, 2500, 4000, 5000,
    ]),
    "upgrade_fee_gap_multiplier": 0.3,
}

agent_setting = {
    "batch_size": 10,
    "room_request_ratio_threshold": 2,
    "padding_rate": 0.4,
    "num_room_multiplier": 0.1,
    "price_multiplier": 0.8,
    "avg_cancel_rate": 0.1,
}

individual_setting = {
    "individual_pop_size": np.array([
        100, 90, 80, 50, 40, 30,
    ]),
    "weekend_rate": np.array([
        0.4, 0.35, 0.3, 0.25, 0.2, 0.1,
    ]),
    "week_rate": np.array([
        0.3, 0.3, 0.2, 0.2, 0.1, 0.05,
    ]),
    "cancel_rate": np.array([
        0.1, 0.08, 0.07, 0.06, 0.05, 0.04,
    ]),
}

# factor range
IND_DEMAND_MUL_SET = (0.5, 1, 2)
STAY_MUL_SET = (1/4, 1/5, 1/7, 1/9)
ROOM_RATE_SET = np.array([
    np.array([
        0.05, 0.3, 0.5, 0.3, 0.2, 0.1,
    ]),
    np.array([
        0.05, 0.1, 0.2, 0.3, 0.3, 0.5,
    ]),
    np.array([
        0.5, 0.3, 0.3, 0.2, 0.1, 0.05,
    ]),
])


config = configparser.ConfigParser(
    interpolation=configparser.ExtendedInterpolation()
)

folder = "settings"
config['META'] = {
    'setting_output_root': folder,
    'factor': "stay duration, room rate, individual demand"
}
config['FACTOR'] = {
    "stay_duration": '\n' + '\t\n'.join([str(v) for v in STAY_MUL_SET]),
    "room_rate": '\n' + '\t\n'.join([str(v) for v in ROOM_RATE_SET]),
    "individual_demand": '\n' + "\t\n".join(
        [str(v) for v in IND_DEMAND_MUL_SET]
    ),
}
with open('meta.ini', 'w') as f:
    config.write(f)

Path(folder).mkdir(parents=True, exist_ok=True)
Path(join(folder, "agent")).mkdir(parents=True, exist_ok=True)
Path(join(folder, "individual")).mkdir(parents=True, exist_ok=True)

with open(join(folder, "base.npy"), 'wb') as f:
    np.save(f, basic_setting)

config = configparser.ConfigParser(
    interpolation=configparser.ExtendedInterpolation()
)
agent_config = configparser.ConfigParser()
ind_config = configparser.ConfigParser()

for ind_demand_mul in IND_DEMAND_MUL_SET:
    for stay_mul in STAY_MUL_SET:
        for room_rate in ROOM_RATE_SET:
            agent_name = (f"stay_{convert_name(stay_mul)}"
                          f"__room_{room_rate.argmax()}")
            ind_name = f"ind_{convert_name(ind_demand_mul)}"
            agent_file_path = join(folder, "agent", agent_name + ".npy")
            if not exists(agent_file_path):
                agent_setting['stay_mul'] = stay_mul
                agent_setting['room_rate'] = room_rate
                with open(agent_file_path, 'wb') as f:
                    np.save(f, agent_setting)
            ind_file_path = join(folder, "individual", ind_name + ".npy")
            if not exists(ind_file_path):
                individual_setting["ind_demand_mul"] = ind_demand_mul
                with open(ind_file_path, 'wb') as f:
                    np.save(f, individual_setting)
            display_name = (
                f"Stay duration: {stay_mul:.2f}| "
                f"Highest room rate: {room_rate.argmax()}| "
                f"Individual demand: {ind_demand_mul:.2f}"
            )
            scenario_name = agent_name + "__" + ind_name
            config[scenario_name] = {
                'ind_demand_mul': ind_demand_mul,
                'stay_mul': stay_mul,
                'room_rate': room_rate,
                'display_name': display_name,
                'agent_setting': agent_file_path,
                "individual_setting": ind_file_path,
                'agent_factor': agent_name,
                'individual_factor': ind_name,
            }
            agent_config[agent_name] = {
                'stay_mul': stay_mul,
                'room_rate': room_rate,
                'path': agent_file_path
            }
    ind_config[ind_name] = {
        'ind_demand_mul': ind_demand_mul,
        'path': ind_file_path
    }
with open(join(folder, 'ind_levels.ini'), "w") as f:
    ind_config.write(f)
with open(join(folder, 'agent_levels.ini'), "w") as f:
    agent_config.write(f)
with open('scenarios.ini', 'w') as f:
    config.write(f)
