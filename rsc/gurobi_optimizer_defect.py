import pandas as pd
import numpy as np

from gurobipy import Model, GRB, quicksum
from itertools import product
from scipy.stats import binom

def id_to_val(id):
    """
    This is only for individual demand and cancel.
    """
    # since id starts from 1
    # FIXME check quantity is larger than id by 1
    return int(id) - 1

def get_order_cancellation(agent_cancel_outcome, agent_cancel_id, order_id):
    return agent_cancel_outcome[int(agent_cancel_id) - 1][int(order_id) -1]

def get_agent_prob(agent_cancel_outcome, agent_cancel_id, agent_cancel_rate):
    agent_cancel_rate = sorted(agent_cancel_rate.items(), 
                               key=lambda x: int(x[0]))
    agent_cancel_rate = np.array([v for i, v in agent_cancel_rate])
    cancel_event = agent_cancel_outcome[int(agent_cancel_id) - 1]
    mask = np.concatenate([cancel_event, (1 - cancel_event)])
    prob = np.concatenate([
        agent_cancel_rate * cancel_event,
        (1 - agent_cancel_rate) * (1 - cancel_event)
    ])
    return np.prod(prob, where=mask.astype(bool))

def get_ind_cancel_prob(cancel_rate, ind_demand_id, ind_cancel_id):
    return binom.pmf(int(ind_cancel_id)-1, int(ind_demand_id)-1, cancel_rate)

# FIXME class contruct
def solve(data_reader, instance_id, upgrade_rule):
    (agent_order_set, time_span, agent_order_price, agent_order_room_quantity,
        agent_order_stay, agent_cancel_rate) = \
            data_reader.collect_agent_info(instance_id)
    (room_type_set, upgrade_levels, downgrade_levels, room_capacity,
     upgrade_fee) = data_reader.collect_hotel_info(upgrade_rule)
    (individual_demand_pmf, individual_room_price, demand_ub,
     individual_cancel_rate) = data_reader.collect_individual_info()
    # maybe use itertools would be more clean
    upgrade_indice = [
        (k, i, j)
        for k in agent_order_set
        for i in agent_order_room_quantity[k]
        for j in upgrade_levels[i]
    ]
    ind_demand_cancel_indice_by_room_time = {
        (room_type, t): [
            (demand_id, cancel_id)
            for demand_id in (np.arange((demand_ub[room_type] + 1)) + 1).astype(str)
            for cancel_id in (np.arange(int(demand_id)) + 1).astype(str)
        ]
        for room_type in room_type_set
        for t in time_span 
    }
    ind_realization_indice = [
        (room_type, t, demand_id, cancel_id)
        for room_type in room_type_set
        for t in time_span
        for demand_id, cancel_id in 
        ind_demand_cancel_indice_by_room_time[room_type, t]
    ]
    room_time_indice = list(product(room_type_set, time_span))
    # creating 2**num_order by numpy would exceed memory
    agent_cancel_indice = (np.arange(2 ** len(agent_order_set)) + 1).astype(str)
    # lazy and add variables accordingly
    agent_cancel_outcome = np.array(list(product([0, 1], 
                                         repeat=len(agent_order_set))))
    agent_order_realization_indice = list(product(agent_cancel_indice, 
                                                  agent_order_set))
    realization_indice = np.hstack([
        np.repeat(ind_realization_indice, repeats=len(agent_cancel_indice), axis=0),
        np.resize(agent_cancel_indice, 
                  (len(ind_realization_indice) * len(agent_cancel_indice), 1))
    ])
    realization_indice = list(map(tuple, realization_indice))

    model = Model("hotel_booking")
    order_acceptance = model.addVars(agent_order_set, vtype=GRB.BINARY,
                                     name=f'Order acceptance')
    upgrade_amount = model.addVars(upgrade_indice, vtype=GRB.INTEGER,
                                   name=f'upgrade amount')
    effective_sale_for_individual = model.addVars(ind_realization_indice,
                                                  vtype=GRB.INTEGER,
                                                  name=f'Sale for individuals')
    capacity_reservation = model.addVars(
        room_time_indice,
        vtype=GRB.INTEGER,
        name='Capacity reserved for individual'
    )
    agent_order_realization = model.addVars(agent_order_realization_indice, 
                                            vtype=GRB.BINARY,
                                            name='Realization of agent orders')
    compensation_room_amount = model.addVars(realization_indice, 
                                             vtype=GRB.INTEGER, lb=0, 
                                             name='Room amount to compensate')

    # no indirect upgrades
    model.addConstrs(
        (
            quicksum(upgrade_amount[order_id, room_type, up_type]
                    for up_type in upgrade_levels[room_type]) <=
            (agent_order_room_quantity[order_id][room_type] *
            order_acceptance[order_id])
            for order_id in agent_order_set
            for room_type in agent_order_room_quantity[order_id]
        ),
        name="Direct upgrades"
    )

    # define individual effective and realized sale
    model.addConstrs(
        (
            effective_sale_for_individual[
                room_type, t, demand_id, cancel_id
            ] <= id_to_val(demand_id) - id_to_val(cancel_id)
            for room_type, t, demand_id, cancel_id in ind_realization_indice
        ),
        name="individual realization by arrivals"
    )

    model.addConstrs(
        (
            effective_sale_for_individual[
                room_type, t, demand_id, cancel_id
            ] <= capacity_reservation[room_type, t]
            for room_type, t, demand_id, cancel_id in ind_realization_indice
        ),
        name="individual realization by conserved capacity"
    )

    # define agent order realization
    model.addConstrs(
        (
            agent_order_realization[agent_cancel_id, order_id] <=
            order_acceptance[order_id]
            for agent_cancel_id, order_id in agent_order_realization_indice
        ),
        name='define order realization by acceptance'
    )

    model.addConstrs(
        (
            agent_order_realization[agent_cancel_id, order_id] <=
            1 - get_order_cancellation(agent_cancel_outcome, agent_cancel_id, 
                                       order_id)
            for agent_cancel_id, order_id in agent_order_realization_indice
        ),
        name='define order realization by cancellation'
    )

    # compensation_amount
    model.addConstrs(
        (
            compensation_room_amount[room_type, t, ind_demand_id, ind_cancel_id,
                                     agent_cancel_id] >=
            - room_capacity[room_type]  
            + effective_sale_for_individual[room_type, t, ind_demand_id, 
                                            ind_cancel_id]
            + quicksum(
                agent_order_stay[order_id][t] * 
                agent_order_room_quantity[order_id][room_type] *
                agent_order_realization[agent_cancel_id, order_id]
                for order_id in agent_order_set
            )
            - quicksum(
                agent_order_stay[order_id][t] * 
                upgrade_amount[order_id, room_type, up_type] *
                (1 - get_order_cancellation(agent_cancel_outcome, 
                                            agent_cancel_id, order_id))
                for order_id in agent_order_set
                for up_type in upgrade_levels[room_type] 
            )
            + quicksum(
                agent_order_stay[order_id][t] * 
                upgrade_amount[order_id, original_type, room_type] *
                (1 - get_order_cancellation(agent_cancel_outcome, 
                                            agent_cancel_id, order_id))
                for order_id in agent_order_set
                for original_type in downgrade_levels[room_type] 
            )
            for room_type, t, ind_demand_id, ind_cancel_id, agent_cancel_id in 
            realization_indice
        ),
        name="number of rooms for compensation"
    )


    model.setObjective(
        quicksum(
            get_agent_prob(agent_cancel_outcome, agent_cancel_id, 
                            agent_cancel_rate) * 
            quicksum(
                agent_order_price[order_id] * 
                agent_order_realization[agent_cancel_id, order_id] + 
                quicksum(
                    agent_order_stay[order_id][t] * quicksum(
                        upgrade_fee[room_type][up_type] * 
                        upgrade_amount[order_id, room_type, up_type] *
                        (1 - get_order_cancellation(agent_cancel_outcome, 
                                                    agent_cancel_id, order_id))
                        for up_type in upgrade_levels[room_type] 
                    )
                    + quicksum(
                        individual_demand_pmf[room_type][t][ind_demand_id]['prob'] * 
                        get_ind_cancel_prob(individual_cancel_rate[room_type],
                                            ind_demand_id, 
                                            ind_cancel_id) *
                        (
                            individual_room_price[room_type] * 
                            effective_sale_for_individual[room_type, t, 
                                                          ind_demand_id,
                                                          ind_cancel_id] -
                            # FIXME compensation is NONE
                            compensation_price[int(room_type)-1][int(t)-1] *
                            compensation_room_amount[room_type, t, 
                                                     ind_demand_id,
                                                     ind_cancel_id,
                                                     agent_cancel_id]
                        )
                        for ind_demand_id, ind_cancel_id in 
                        ind_demand_cancel_indice_by_room_time[room_type, t] 
                    )
                    for room_type in room_type_set
                    for t in time_span
                )
                for order_id in agent_order_set
            )             
            for agent_cancel_id in agent_cancel_indice   
        )
        ,
        GRB.MAXIMIZE
    )
    # FIXME DUPLICATE
    # TODO outcome id range could be narrower.
    # maybe using itertools would be cleaner
    # or it require B_i parameter to pass
    model.Params.TimeLimit = float('inf')
    # model.Params.TimeLimit = 10
    model.Params.MIPGap = 0
    model.optimize()

    # print("Objective value:", model.objVal)
    # print("Runtime: ", model.Runtime)

    # for order_id in agent_order_set:
    #     print("Order {}: {}".format(
    #         order_id, acc_verbose(order_acceptance[order_id].x)))
    #     for room_type in room_type_set:
    #         if agent_order_room_quantity[order_id][room_type] != 0:
    #             print('\tRoom type %s: %s' %
    #                     (room_type, agent_order_room_quantity[order_id][room_type]),)
    #             for up_type in upgrade_levels[room_type]:
    #                 if upgrade_amount[order_id, room_type, up_type].x != 0:
    #                     print('\t\tUpgrade to type %s: %d'% (up_type, upgrade_amount[order_id, room_type, up_type].x))
    #     print('=' * 20)

    # for room_type in room_type_set:
    #     print(room_type)
    #     for outcome_id in individual_demand_pmf[room_type]:
    #         print(outcome_id, ':', individual_demand_pmf[room_type][outcome_id],)
    #         for t in time_span:
    #             print(effective_sale_for_individual[room_type, t, outcome_id].x, end=', ')
    #         print()
    #     print('='*20)

    # model.write('mip1.rlp')

    up_result = np.zeros(
        (len(agent_order_set), len(room_type_set), len(room_type_set))
    )
    for indice in upgrade_indice:
        up_result[int(indice[0]) -1, int(indice[1]) -1, int(indice[2]) -1] = \
            upgrade_amount[indice].x
    sale = pd.DataFrame.from_dict(
        model.getAttr('x', effective_sale_for_individual),
        orient="index"
    )
    mul_index = pd.MultiIndex.from_tuples(
        effective_sale_for_individual.keys(),
        names=["room", "time", "demand", 'cancel']
    )
    sale = sale.reindex(mul_index)
    sale.columns = ["sale"]
    acc_result = np.array(
        [order_acceptance[str(i + 1)].x for i in range(len(agent_order_set))]
    )
    return acc_result, up_result, model.objVal, sale