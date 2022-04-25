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

def get_ind_cancel_prob(cancel_rate, ind_reservation_id, ind_cancel_val):
    if type(ind_cancel_val) == int:
        return binom.pmf(
            ind_cancel_val, 
            id_to_val(ind_reservation_id), 
            cancel_rate
        )
    else:
        raise Exception("wrong type")

# FIXME class contruct
def solve(data_reader, instance_id, upgrade_rule):
    (agent_order_set, time_span, agent_order_price, agent_order_room_quantity,
        agent_order_stay, agent_cancel_rate) = \
            data_reader.collect_agent_info(instance_id)
    (room_type_set, upgrade_levels, downgrade_levels, room_capacity,
     upgrade_fee, compensation_price) = \
        data_reader.collect_hotel_info(upgrade_rule)
    (individual_demand_pmf, individual_room_price, demand_ub,
     individual_cancel_rate) = data_reader.collect_individual_info()

    ### indice
    # TODO maybe use itertools would be more clean
    upgrade_indice = [
        (k, i, j)
        for k in agent_order_set
        for i in room_type_set
        for j in upgrade_levels[i]
    ]
    # FIXME rely the relationship between ID and quantity
    ind_demand_indice = [
        (room_type, t, demand_id)
        for room_type in room_type_set
        for t in time_span
        for demand_id in (np.arange((demand_ub[room_type] + 1)) + 1).astype(str)
    ]
    ind_demand_reservation_indice = [
        (room_type, t, demand_id, reservation_id)
        for room_type in room_type_set
        for t in time_span
        for demand_id in (np.arange((demand_ub[room_type] + 1)) + 1).astype(str)
        for reservation_id in (np.arange(int(demand_id)) + 1).astype(str)
    ]

    ### build model
    model = Model("hotel_booking")
    ### add variables
    order_acceptance = model.addVars(agent_order_set, vtype=GRB.BINARY,
                                     name=f'Order acceptance')
    upgrade_amount = model.addVars(upgrade_indice, vtype=GRB.CONTINUOUS,
                                   name=f'upgrade amount')
    individual_reservation = model.addVars(ind_demand_indice,
                                           vtype=GRB.CONTINUOUS,
                                           name=f'Sale for individuals')
    individual_realization = model.addVars(ind_demand_reservation_indice,
                                           vtype=GRB.CONTINUOUS,
                                           name=f'Sale for individuals')
    is_valid_reservation = model.addVars(ind_demand_reservation_indice, 
                                         vtype=GRB.BINARY,  
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
    # agent consumption can not exceed capacity
    model.addConstrs(
        (
            quicksum(
                agent_order_stay[order_id][t] * 
                agent_order_room_quantity[order_id][room_type] *
                order_acceptance[order_id]
                for order_id in agent_order_set
            )
            - quicksum(
                agent_order_stay[order_id][t] * 
                upgrade_amount[order_id, room_type, up_type] 
                for order_id in agent_order_set
                for up_type in upgrade_levels[room_type] 
            )
            + quicksum(
                agent_order_stay[order_id][t] * 
                upgrade_amount[order_id, original_type, room_type]
                for order_id in agent_order_set
                for original_type in downgrade_levels[room_type] 
            ) <= 
            room_capacity[room_type]
            for room_type in room_type_set
            for t in time_span
        ),
        name='agent order capacity'
    )
    # define individual reservation by demand
    model.addConstrs(
        (
            individual_reservation[
                room_type, t, demand_id
            ] <= id_to_val(demand_id)
            for room_type, t, demand_id in ind_demand_indice
        ),
        name="individual reservation by demand"
    )
    # define individual reservation by left vacancy
    model.addConstrs(
        (
            individual_reservation[
                room_type, t, demand_id
            ] <= room_capacity[room_type] 
            - quicksum(
                agent_order_stay[order_id][t] * 
                agent_order_room_quantity[order_id][room_type] *
                order_acceptance[order_id]
                for order_id in agent_order_set
            )
            + quicksum(
                agent_order_stay[order_id][t] * 
                upgrade_amount[order_id, room_type, up_type] 
                for order_id in agent_order_set
                for up_type in upgrade_levels[room_type] 
            )
            - quicksum(
                agent_order_stay[order_id][t] * 
                upgrade_amount[order_id, original_type, room_type]
                for order_id in agent_order_set
                for original_type in downgrade_levels[room_type] 
            )
            for room_type, t, demand_id in ind_demand_indice
        ),
        name="individual reservation by demand"
    )
    # verify the reservation is valid
    model.addConstrs(
        (
            (1 - is_valid_reservation[room_type, t, demand_id, reservation_id])
            * id_to_val(demand_id) >=
            id_to_val(reservation_id) 
            - individual_reservation[room_type, t, demand_id]
            for room_type, t, demand_id, reservation_id in 
            ind_demand_reservation_indice
        ),
        name='reservation verification overflow'
    )
    # verify the reservation is valid
    model.addConstrs(
        (
            (is_valid_reservation[room_type, t, demand_id, reservation_id] - 1)
            * id_to_val(demand_id) <=
            id_to_val(reservation_id) 
            - individual_reservation[room_type, t, demand_id]
            for room_type, t, demand_id, reservation_id in 
            ind_demand_reservation_indice
        ),
        name='reservation verification underflow'
    )
    # set invalid individual realization to zero
    model.addConstrs(
        (
            individual_realization[room_type, t, demand_id, reservation_id] <= 
            is_valid_reservation[room_type, t, demand_id, reservation_id] * 
            id_to_val(reservation_id) * individual_room_price[room_type]
            for room_type, t, demand_id, reservation_id in 
            ind_demand_reservation_indice
        ),
        name='individual realization filter'
    ) 
    # limit valid individual realization by reservation and withdrawl
    model.addConstrs(
        (
            individual_realization[room_type, t, demand_id, reservation_id] <= 
            quicksum(
                get_ind_cancel_prob(individual_cancel_rate[room_type],
                                    reservation_id, cancel_val) *
                (id_to_val(reservation_id) - cancel_val) *
                individual_room_price[room_type]
                for cancel_val in range(int(reservation_id))
            )
            for room_type, t, demand_id, reservation_id in 
            ind_demand_reservation_indice
        ),
        name='individual realization UB'
    ) 

    # objective function
    model.setObjective(
        quicksum(
            agent_order_price[order_id] * order_acceptance[order_id]
            for order_id in agent_order_set
        ) +
        quicksum(
            agent_order_stay[order_id][t] * (
                quicksum(
                    upgrade_fee[room_type][up_type] * 
                    upgrade_amount[order_id, room_type, up_type]
                    for room_type in room_type_set
                    for up_type in upgrade_levels[room_type]
                )
            )
            for order_id in agent_order_set
            for t in time_span
        ) +
        quicksum(
            individual_demand_pmf[room_type][t][demand_id]['prob'] *
            individual_realization[room_type, t, demand_id, reservation_id]
            for room_type, t, demand_id, reservation_id in 
            ind_demand_reservation_indice
        ),
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
        model.getAttr('x', individual_realization),
        orient="index"
    )
    mul_index = pd.MultiIndex.from_tuples(
        individual_realization.keys(),
        names=["room", "time", "demand", 'cancel']
    )
    sale = sale.reindex(mul_index)
    sale.columns = ["sale"]
    acc_result = np.array(
        [order_acceptance[str(i + 1)].x for i in range(len(agent_order_set))]
    )
    return acc_result, up_result, model.objVal, sale