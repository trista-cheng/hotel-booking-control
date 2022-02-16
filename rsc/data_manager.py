import copy
import json
import numpy as np
import pandas as pd

from pathlib import Path
from os.path import join

from data_generator import DataGenerator


OUTPUT_ROOT = "data"
REPLICATE_NUM = 10

# static setting
TIME_SPAN_LEN = 21
NUM_ROOM_TYPE = 6
NUM_ROOM_MULTIPLIER = 0.05
PRICE_MULTIPLIER = 0.8
UPGRADE_FEE_MULTIPLIER = 0.8
PADDING_RATE = 0.2
CAPACITY = np.array([200, 150, 100, 70, 30, 10])
INDIVIDUAL_PRICE = np.array([50, 60, 100, 180, 200, 250])
INDIVIDUAL_POP_SIZE = np.array([200, 150, 150, 100, 50, 50])
WEEKEND_RATE = np.array([0.5, 0.3, 0.3, 0.2, 0.2, 0.1])
WEEK_RATE = np.array([0.4, 0.2, 0.2, 0.1, 0.1, 0.1])

# factor range
IND_DEMAND_MUL_SET = (0.5, 1, 2)
STAY_MUL_SET = (1/TIME_SPAN_LEN, 1/10, 1/5, 1/2)
CAPACITY_MUL_SET = [[1, 1, 1, 1, 1, 1]]
all_capacity = CAPACITY.sum()
for c_id, c in enumerate(CAPACITY):
    down_mul = 1 - (c / (all_capacity - c))
    mul = np.repeat(down_mul, len(CAPACITY))
    mul[c_id] = 2
    CAPACITY_MUL_SET.append(mul)
CAPACITY_MUL_SET = np.array(CAPACITY_MUL_SET)


def save(data: np.array, name: str, instance_id: int, scenario=None):
    """
    Not to build folder if name is None
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
    def __init__(self, data_gen, replicate_num, time_span_len, num_room_type,
                 capacity, individual_price, individual_pop_size,
                 original_ind_success_rate, num_room_multiplier,
                 price_multiplier, upgrade_fee_multiplier, padding_rate):

        # register the attributes which are not factors.
        self.data_gen = data_gen
        self.replicate_num = replicate_num
        self.time_span_len = time_span_len
        self.num_room_type = num_room_type
        self.capacity = capacity
        self.individual_price = individual_price
        self.individual_pop_size = individual_pop_size
        self.original_ind_success_rate = original_ind_success_rate
        self.num_room_multiplier = num_room_multiplier
        self.price_multiplier = price_multiplier
        self.upgrade_fee_multiplier = upgrade_fee_multiplier
        self.padding_rate = padding_rate

    def save_agent(self, stay_mul, capacity_mul):
        avg_stay_duration = np.max(int(self.time_span_len * stay_mul), 0)
        # FIXME 5 is magic number
        num_order = int((self.time_span_len / avg_stay_duration) * 5)
        avg_num_room = self.num_room_multiplier * self.capacity
        scenario = f"stay_{stay_mul}_twicecapacity_{np.argmax(capacity_mul)}"
        for i in range(self.replicate_num):
            agent_order_price, agent_order_room_quantity, agent_order_stay, \
                upgrade_fee = self.data_gen.generate_agent_order(
                avg_stay_duration, num_order, avg_num_room, self.padding_rate,
                capacity_mul, self.price_multiplier, self.upgrade_fee_multiplier
            )
            save(agent_order_price, "agent_order_price", i, scenario)
            save(agent_order_room_quantity, "agent_order_room_quantity", i,
                 scenario)
            save(agent_order_stay, "agent_order_stay", i, scenario)
            save(upgrade_fee, "upgrade_fee", i, scenario)

    def save_individual(self, ind_demand_mul):
        individual_success_rate = (
            ind_demand_mul * self.original_ind_success_rate
        )
        scenario = f"ind_demand_{ind_demand_mul}"
        for i in range(self.replicate_num):
            folder = join(OUTPUT_ROOT, "individual_demand_prob", scenario)
            Path(folder).mkdir(parents=True, exist_ok=True)
            file_name = join(folder, str(i))
            pmf = self.data_gen.generate_individual(
                individual_success_rate, self.individual_pop_size, file_name
            )
            # FIXME 100 is magic number
            pmf_dict = {}
            for room_id in range(1, self.num_room_type + 1):
                pmf_dict[room_id] = {}
                for time_id in range(1, self.time_span_len + 1):
                    pmf_dict[room_id][time_id] = {}
                    for s in range(1, 101):
                        pmf_dict[room_id][time_id][s] = pmf[(room_id, time_id, s)]
            folder = join(OUTPUT_ROOT, "individual_demand_pmf", scenario)
            Path(folder).mkdir(parents=True, exist_ok=True)
            with open(join(folder, f"{i}.json"), "w", encoding='utf-8') as f:
                json.dump(pmf_dict, f, ensure_ascii=False, indent=4)



class DataManager:

    def __init__(self, replicate_num, time_span_len, num_room_type, capacity,
                 individual_price, individual_pop_size, week_rate, weekend_rate,
                 num_room_multiplier, price_multiplier,
                 upgrade_fee_multiplier) -> None:

        # TODO FEW flexibility in individual success rate.
        is_weekend = np.resize([0, 0, 0, 0, 0, 1, 1], time_span_len)
        week_msk = 1 - is_weekend
        original_ind_success_rate = week_rate.reshape((-1, 1)) * week_msk
        original_ind_success_rate += weekend_rate.reshape((-1, 1)) * is_weekend
        self.original_ind_success_rate = original_ind_success_rate

        factor_manager = FactorManager(
            DataGenerator(
                time_span_len, capacity, individual_price
            ), replicate_num, time_span_len, num_room_type, capacity,
            individual_price, individual_pop_size,
            self.original_ind_success_rate, num_room_multiplier,
            price_multiplier, upgrade_fee_multiplier, PADDING_RATE
        )
        self.factor_manager = factor_manager

    def simulate(self, ind_demand_mul_set, stay_mul_set, capacity_mul_set):
        ## Save static data
        save(CAPACITY, "capacity", None)
        save(INDIVIDUAL_PRICE, "individual_room_price", None)

        ## Generate data for each factor
        for stay_mul in stay_mul_set:
            for capacity_mul in capacity_mul_set:
                factor_manager = copy.deepcopy(self.factor_manager)
                factor_manager.save_agent(stay_mul, capacity_mul)
        for ind_demand_mul in ind_demand_mul_set:
            factor_manager = copy.deepcopy(self.factor_manager)
            factor_manager.save_individual(ind_demand_mul)


data_manager = DataManager(REPLICATE_NUM, TIME_SPAN_LEN, NUM_ROOM_TYPE,
    CAPACITY, INDIVIDUAL_PRICE, INDIVIDUAL_POP_SIZE, WEEK_RATE,
    WEEKEND_RATE, NUM_ROOM_MULTIPLIER, PRICE_MULTIPLIER, UPGRADE_FEE_MULTIPLIER
)

data_manager.simulate(IND_DEMAND_MUL_SET, STAY_MUL_SET, CAPACITY_MUL_SET)

