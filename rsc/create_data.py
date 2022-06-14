import json
import configparser
import numpy as np
import shutil
import pandas as pd

from pathlib import Path
from os.path import join, exists
from collections import defaultdict

from data_generator import DataGenerator
from tools import clean_archive_output

OUTPUT_ROOT = "data"

def save(data: np.array, entity_name: str, factor_level=None, instance_id=None,
         output_root=OUTPUT_ROOT):
    """
    Parameters
    ------------
    data(np.array)
    entity_name: used to create a folder titled as `enttiy_name`
    factor_level(str or None): used to create a nested folder to label the
        certain factor levels
    instance_id: used as file name when the data is under certain factor level

    Not to build folder if `instance_id` and `factor_level` is None
    """
    if len(data.shape) == 1:
        df = pd.DataFrame(data, columns=["value"])
        df.index += 1
        content = df.to_dict()['value']
    else:
        df = pd.DataFrame(data, columns=range(1, data.shape[1] + 1))
        df.index += 1
        content = df.T.to_dict()

    if (instance_id != None) & (factor_level != None):
        folder = join(output_root, entity_name, factor_level)
        Path(folder).mkdir(parents=True, exist_ok=True)
        df.to_csv(join(folder, f'{instance_id}.csv'), index=False)
        with open(join(folder, f"{instance_id}.json"), "w",
                  encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=4)
    else:
        Path(join(output_root)).mkdir(parents=True, exist_ok=True)
        df.to_csv(join(output_root, f'{entity_name}.csv'), index=False)
        with open(join(output_root, f"{entity_name}.json"), "w",
                  encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=4)

class FactorManager:
    def __init__(self, data_gen, replicate_num):

        # register attributes that are not factors.
        self.data_gen = data_gen
        self.replicate_num = replicate_num

    def save_agent(self, agent_level, agent_setting):
        avg_stay_duration = np.max(
            [int(self.data_gen.time_span_len * agent_setting["stay_mul"]), 1]
        )
        avg_num_room = (agent_setting['num_room_multiplier'] *
                        self.data_gen.capacity)
        # FIXME these setting aboove calculated here may be BAD, should be in
        # data generator
        agent_setting['avg_stay_duration'] = avg_stay_duration
        agent_setting['avg_num_room'] = avg_num_room
        for i in range(self.replicate_num):
            (agent_order_price, agent_order_room_quantity, agent_order_stay,
             agent_cancel_rate) = \
                self.data_gen.generate_agent_order(**agent_setting)
            save(agent_order_price, "agent_order_price", agent_level, i)
            save(agent_order_room_quantity, "agent_order_room_quantity",
                 agent_level, i)
            save(agent_order_stay, "agent_order_stay", agent_level, i)
            save(agent_cancel_rate, "agent_cancel_rate", agent_level, i)


    def save_individual(self, ind_level, ind_setting):
        is_weekend = np.resize([0, 0, 0, 0, 0, 1, 1],
                               self.data_gen.time_span_len)
        week_msk = 1 - is_weekend
        original_ind_success_rate = (ind_setting['week_rate'].reshape((-1, 1)) *
                                     week_msk)
        original_ind_success_rate += (
            ind_setting['weekend_rate'].reshape((-1, 1)) * is_weekend
        )
        individual_success_rate = (
            ind_setting['ind_demand_mul'] * original_ind_success_rate
        )
        ind_setting['individual_success_rate'] = individual_success_rate
        # FIXME not sure the calculation should be here
        pmf_dict_tuple_key, demand_ub, cancel_rate = \
            self.data_gen.generate_individual(
            **ind_setting
        )
        # TODO decide to store as numpy(tuple key) or json(nested key)
        pmf_dict_nested_key = defaultdict(dict)
        for room_id, time_id, outcome_id in pmf_dict_tuple_key:
            pmf_dict_nested_key[room_id] = pmf_dict_nested_key.get(room_id, {})
            pmf_dict_nested_key[room_id][time_id] = \
                pmf_dict_nested_key[room_id].get(time_id, {})
            pmf_dict_nested_key[room_id][time_id][outcome_id] = \
                pmf_dict_nested_key[room_id][time_id].get(outcome_id, {})
            pmf_dict_nested_key[room_id][time_id][outcome_id] = \
                pmf_dict_tuple_key[(room_id, time_id, outcome_id)]
        # FIXME not sure how to save as csv by dict
        # pmf_df = pd.DataFrame.from_dict(pmf_dict_tuple_key, orient='index')
        folder = join(OUTPUT_ROOT, "individual_demand_pmf")
        Path(folder).mkdir(parents=True, exist_ok=True)
        with open(join(folder, f"{ind_level}.json"), "w",
                  encoding='utf-8') as f:
            json.dump(pmf_dict_nested_key, f, ensure_ascii=False, indent=4)
        with open(join(folder, f"{ind_level}.npy"), "wb") as f:
            np.save(f, pmf_dict_tuple_key)
        # TODO demand ub may be save by hotel info
        save(demand_ub, "demand_ub")
        save(cancel_rate, 'individual_cancel_rate')

class DataManager:

    def __init__(self, base, agent_levels, ind_levels) -> None:
        self.base = base
        self.agent_levels = agent_levels
        self.ind_levels = ind_levels

    def simulate(self, replicate_num)-> None:
        ## Generate basic hotel info
        data_generator = DataGenerator(**base)
        capacity, individual_room_price, upgrade_fee, compensation_price = \
            data_generator.generate_hotel_info()
        save(capacity, "capacity")
        save(individual_room_price, "individual_room_price")
        save(upgrade_fee, "upgrade_fee")
        save(compensation_price, 'compensation_price')

        factor_manager = FactorManager(data_generator, replicate_num)
        # generate agent
        for agent_level in agent_levels.sections():
            with open(agent_levels[agent_level]["path"], "rb") as f:
                agent_setting = np.load(f, allow_pickle=True)[()]
            factor_manager.save_agent(agent_level, agent_setting)
        # generate individual
        for ind_level in ind_levels.sections():
            with open(ind_levels[ind_level]["path"], "rb") as f:
                ind_setting = np.load(f, allow_pickle=True)[()]
            factor_manager.save_individual(ind_level, ind_setting)


SETTING_ROOT = 'settings'
REPLICATE_NUM = 30
# FIXME replicate num may be in meta.ini
if __name__ == "__main__":
    clean_archive_output(['data'])
    with open(join(SETTING_ROOT, 'base.npy'), 'rb') as f:
        base = np.load(f, allow_pickle=True)[()]
    agent_levels = configparser.ConfigParser()
    agent_levels.read(join(SETTING_ROOT, 'agent_levels.ini'))
    ind_levels = configparser.ConfigParser()
    ind_levels.read(join(SETTING_ROOT, 'ind_levels.ini'))

    data_manager = DataManager(base, agent_levels, ind_levels)
    data_manager.simulate(replicate_num=REPLICATE_NUM)
