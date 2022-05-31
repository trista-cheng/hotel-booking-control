import shutil
import numpy as np

from os.path import exists
from data_reader import CSVDataReader

def clean_archive_output(archive_folders):
    for archive_folder in archive_folders:
        if exists(archive_folder):
            shutil.rmtree(archive_folder)


def get_ind_exp_req(scenario: dict, time_span_len: int, with_ind_cancel=True):
    # generate success rate
    with open(scenario['individual_setting'], 'rb') as f:
        ind_setting = np.load(f, allow_pickle=True)[()]
    is_weekend = np.resize([0, 0, 0, 0, 0, 1, 1], time_span_len)
    week_msk = 1 - is_weekend
    original_ind_success_rate = (ind_setting['week_rate'].reshape((-1, 1)) *
                                 week_msk)
    original_ind_success_rate += (
        ind_setting['weekend_rate'].reshape((-1, 1)) * is_weekend
    )
    individual_success_rate = (
        ind_setting['ind_demand_mul'] * original_ind_success_rate
    )
    # FIXME only allow with_ind_cancel
    ind_exp_req = (
        individual_success_rate *
        ind_setting['individual_pop_size'].reshape((-1, 1)) *
        (1 - (ind_setting['cancel_rate'] * with_ind_cancel).reshape((-1, 1)))
    )
    return ind_exp_req

def get_exp_ub(scenario, instance_id, upgrade_rule, with_agent_cancel):
    reader = CSVDataReader(scenario=scenario)
    (room_type_set, room_capacity, upgrade_fee, compensation_price) = \
        reader.collect_hotel_info(upgrade_rule=upgrade_rule)
    (agent_order_set, time_span, agent_order_price, agent_order_room_quantity,
     agent_order_stay, agent_cancel_rate) = \
        reader.collect_agent_info(instance_id=instance_id)
    (individual_demand_pmf, individual_room_price, demand_ub,
     individual_cancel_rate) = reader.collect_individual_info()
    if with_agent_cancel:
        agent_ub = (agent_order_price * (1 - agent_cancel_rate)).sum()
        agent_req = np.dot(
            agent_order_room_quantity.T,
            agent_order_stay * (1 - agent_cancel_rate).reshape((-1, 1))
        )
    else:
        agent_ub = agent_order_price.sum()
        agent_req = np.dot(agent_order_room_quantity.T, agent_order_stay)
    ind_exp_req = get_ind_exp_req(scenario, len(time_span), demand_ub,
                                  individual_cancel_rate)
    ind_ub = (ind_exp_req * individual_room_price.reshape((-1, 1))).sum()

    capacity_value = (((ind_ub + agent_ub) / (ind_exp_req + agent_req).sum()) *
                      (room_capacity.sum() * len(time_span)))

    return agent_ub, ind_ub, capacity_value
