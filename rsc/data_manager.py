import copy
import json
import numpy as np
import pandas as pd

from pathlib import Path
from os.path import join

from data_generator import DataGenerator


OUTPUT_ROOT = "data"
REPLICATE_NUM = 10
BATCH_SIZE = 10

# static setting
TIME_SPAN_LEN = 21
NUM_ROOM_TYPE = 6
NUM_ROOM_MULTIPLIER = 0.1
PRICE_MULTIPLIER = 0.8
UPGRADE_FEE_MULTIPLIER = 0.8 * 0.6
PADDING_RATE = 0.3
CAPACITY = np.array([200, 150, 100, 70, 60, 50])
INDIVIDUAL_PRICE = np.array([90, 100, 130, 140, 180, 200])
INDIVIDUAL_POP_SIZE = np.array([100, 80, 70, 50, 40, 20])
WEEKEND_RATE = np.array([0.4, 0.3, 0.3, 0.2, 0.2, 0.1])
WEEK_RATE = np.array([0.2, 0.15, 0.15, 0.1, 0.1, 0.05])
ROOM_REQUEST_RATIO_THRESHOLD = 1.5  # request quantity for each type must exceed times 
# of capacity

# factor range
IND_DEMAND_MUL_SET = (0.5, 1, 2)
STAY_MUL_SET = (1/4, 1/5, 1/7, 1/9)
ROOM_RATE_SET = np.array([
    np.array([0.1, 0.3, 0.5, 0.4, 0.2, 0.05]),
    np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.05]),
    np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5]),
])


def save(data: np.array, name: str, instance_id: int, scenario=None):
    # FIXME scenario 名字修掉
    """
    Not to build folder if `instance_id` is None
    """
    if len(data.shape) == 1:
        df = pd.DataFrame(data, columns=["value"])
        df.index += 1
        content = df.to_dict()['value']
    else:
        df = pd.DataFrame(data, columns=range(1, data.shape[1] + 1))
        df.index += 1
        content = df.T.to_dict()

    if instance_id != None:
        Path(join(OUTPUT_ROOT, name, scenario)).mkdir(parents=True, exist_ok=True)
        df.to_csv(join(OUTPUT_ROOT, name, scenario, f'{instance_id}.csv'), index=False)
        with open(join(OUTPUT_ROOT, name, scenario, f"{instance_id}.json"), "w",
                  encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=4)
    else:
        Path(join(OUTPUT_ROOT)).mkdir(parents=True, exist_ok=True)
        df.to_csv(join(OUTPUT_ROOT, f'{name}.csv'), index=False)
        with open(join(OUTPUT_ROOT, f"{name}.json"), "w",
                  encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=4)

class FactorManager:
    def __init__(self, data_gen, replicate_num, individual_pop_size, 
                 original_ind_success_rate, num_room_multiplier, 
                 room_request_ratio_threshold, price_multiplier,
                 upgrade_fee_multiplier, padding_rate, batch_size):

        # register attributes that are not factors.
        self.data_gen = data_gen
        self.replicate_num = replicate_num
        self.time_span_len = data_gen.time_span_len
        self.num_room_type = data_gen.num_room_type
        self.capacity = data_gen.capacity
        self.individual_pop_size = individual_pop_size
        self.original_ind_success_rate = original_ind_success_rate
        self.num_room_multiplier = num_room_multiplier
        self.room_request_ratio_threshold = room_request_ratio_threshold
        self.price_multiplier = price_multiplier
        self.upgrade_fee_multiplier = upgrade_fee_multiplier
        self.padding_rate = padding_rate
        self.batch_size = batch_size

    def save_agent(self, stay_mul, room_rate, scenario):
        avg_stay_duration = np.max([int(self.time_span_len * stay_mul), 1])
        avg_num_room = self.num_room_multiplier * self.capacity
        for i in range(self.replicate_num):
            agent_order_price, agent_order_room_quantity, agent_order_stay, \
                upgrade_fee = self.data_gen.generate_agent_order(
                    self.room_request_ratio_threshold, avg_stay_duration, 
                    avg_num_room, self.padding_rate, room_rate, 
                    self.price_multiplier, self.upgrade_fee_multiplier, 
                    self.batch_size
                )
            save(agent_order_price, "agent_order_price", i, scenario)
            save(agent_order_room_quantity, "agent_order_room_quantity", i,
                 scenario)
            save(agent_order_stay, "agent_order_stay", i, scenario)
            save(upgrade_fee, "upgrade_fee", None)

    def save_individual(self, ind_demand_mul, scenario):
        individual_success_rate = (
            ind_demand_mul * self.original_ind_success_rate
        )
        # for i in range(self.replicate_num):
        folder = join(OUTPUT_ROOT, "individual_demand_prob")
        Path(folder).mkdir(parents=True, exist_ok=True)
        file_name = join(folder, f"{scenario}")
        pmf, demand_ub = self.data_gen.generate_individual(
            individual_success_rate, self.individual_pop_size, file_name
        )
        pmf_dict = {}
        for room_id in range(1, self.num_room_type + 1):
            pmf_dict[room_id] = {}
            for time_id in range(1, self.time_span_len + 1):
                pmf_dict[room_id][time_id] = {}
                for s in range(1, np.min([self.capacity.max(), self.individual_pop_size.max()]) + 2):
                    pmf_dict[room_id][time_id][s] = pmf[(room_id, time_id, s)]
        folder = join(OUTPUT_ROOT, "individual_demand_pmf")
        Path(folder).mkdir(parents=True, exist_ok=True)
        with open(join(folder, f"{scenario}.json"), "w", encoding='utf-8') as f:
            json.dump(pmf_dict, f, ensure_ascii=False, indent=4)
        save(demand_ub, "demand_ub", None)



class DataManager:

    def __init__(self, replicate_num, time_span_len, capacity, individual_price, 
                 individual_pop_size, week_rate, weekend_rate,
                 num_room_multiplier, room_request_ratio_threshold, 
                 price_multiplier, upgrade_fee_multiplier, 
                 padding_rate, batch_size) -> None:

        # TODO FEW flexibility in individual success rate.
        is_weekend = np.resize([0, 0, 0, 0, 0, 1, 1], time_span_len)
        week_msk = 1 - is_weekend
        original_ind_success_rate = week_rate.reshape((-1, 1)) * week_msk
        original_ind_success_rate += weekend_rate.reshape((-1, 1)) * is_weekend
        self.original_ind_success_rate = original_ind_success_rate.copy()

        factor_manager = FactorManager(
            DataGenerator(
                time_span_len, capacity, individual_price
            ), replicate_num, individual_pop_size,
            self.original_ind_success_rate, num_room_multiplier,
            room_request_ratio_threshold, price_multiplier, 
            upgrade_fee_multiplier, padding_rate, batch_size
        )
        self.factor_manager = factor_manager
        self.scenarios = {}
        ## Save static data
        save(capacity, "capacity", None)
        save(individual_price, "individual_room_price", None)

    def simulate(self, ind_demand_mul_set, stay_mul_set, room_rate_set):
        ## Generate data for each factor
        self.scenarios["agent"] = []
        for stay_mul in stay_mul_set:
            for room_rate in room_rate_set:
                # TODO is it neccessary to deepcopy 
                scenario = f"stay_mul_{stay_mul:.3f}_high_request_room_id_{np.argmax(room_rate)}"
                self.scenarios["agent"].append(scenario)
                factor_manager = copy.deepcopy(self.factor_manager)
                factor_manager.save_agent(stay_mul, room_rate, scenario)
        self.scenarios["individual"] = []
        for ind_demand_mul in ind_demand_mul_set:
            scenario = f"ind_demand_{ind_demand_mul}"
            self.scenarios["individual"].append(scenario)
            factor_manager = copy.deepcopy(self.factor_manager)
            factor_manager.save_individual(ind_demand_mul, scenario)
        
        with open("scenarios.json", "w") as f:
            json.dump(self.scenarios, f)
        

if __name__ == "__main__":
    data_manager = DataManager(REPLICATE_NUM, TIME_SPAN_LEN, CAPACITY, 
        INDIVIDUAL_PRICE, INDIVIDUAL_POP_SIZE, WEEK_RATE, WEEKEND_RATE, 
        NUM_ROOM_MULTIPLIER, ROOM_REQUEST_RATIO_THRESHOLD, PRICE_MULTIPLIER, 
        UPGRADE_FEE_MULTIPLIER, PADDING_RATE, BATCH_SIZE,
    )

    data_manager.simulate(IND_DEMAND_MUL_SET, STAY_MUL_SET, ROOM_RATE_SET)

