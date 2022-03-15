import time
import pandas as pd
import numpy as np

from gurobipy import Model, GRB, quicksum

# FIXME class contruct
def solve(data_reader, instance_id, upgrade_rule):
    (agent_order_set, time_span, agent_order_price, agent_order_room_quantity,
        agent_order_stay) = data_reader.collect_agent_info(instance_id)
    (room_type_set, upgrade_levels, downgrade_levels, room_capacity, 
     upgrade_fee) = data_reader.collect_hotel_info(upgrade_rule)
    individual_demand_pmf, individual_room_price, demand_ub = \
        data_reader.collect_individual_info()

    upgrade_indice = [
        (k, i, j)
        for k in agent_order_set
        for i in agent_order_room_quantity[k]
        for j in upgrade_levels[i]
    ]
    demand_indice = [
        (room_type, t, str(outcome_id))
        for room_type in room_type_set
        for t in time_span
        for outcome_id in range(1, demand_ub[room_type] + 2)
    ]
    # Set population size as UB of maximum possible value B_i
    # TODO outcome id range could be narrower.
    # range(
    #   1, 
    #   np.min([
    #     INDIVIDUAL_POP_SIZE[int(room_type) - 1], 
    #     room_capacity[room_type]
    #   ]) + 2
    # )
    # maybe use itertools would be more clean
    # or it require B_i parameter to pass
    model = Model("hotel_booking")
    order_acceptance = model.addVars(agent_order_set, vtype=GRB.BINARY,
                                     name=f'Order acceptance')
    upgrade_amount = model.addVars(upgrade_indice, vtype=GRB.INTEGER,
                                   name=f'upgrade amount')
    effective_sale_for_individual = model.addVars(demand_indice,
                                                  vtype=GRB.INTEGER,
                                                  name=f'Sale for individuals')

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

    # capacity constraint
    model.addConstrs(
        (
            quicksum(
                agent_order_stay[order_id][t] *
                agent_order_room_quantity[order_id][room_type] *
                order_acceptance[order_id] for order_id in agent_order_set
            ) -
            quicksum(
                agent_order_stay[order_id][t] *
                upgrade_amount[order_id, room_type, up_type]
                for order_id in agent_order_set
                for up_type in upgrade_levels[room_type]
            ) +
            quicksum(
                agent_order_stay[order_id][t] *
                upgrade_amount[order_id, low_type, room_type]
                for order_id in agent_order_set
                for low_type in downgrade_levels[room_type]
            )
            <= room_capacity[room_type]
            for t in time_span for room_type in room_type_set
        ),
        name="Capacity"
    )

    # define the effective sale for individual
    model.addConstrs((
        room_capacity[room_type] -
        quicksum(
            agent_order_stay[order_id][t] *
            agent_order_room_quantity[order_id][room_type] *
            order_acceptance[order_id] for order_id in agent_order_set
        ) +
        quicksum(
            agent_order_stay[order_id][t] *
            upgrade_amount[order_id, room_type, up_type]
            for order_id in agent_order_set
            for up_type in upgrade_levels[room_type]
        ) -
        quicksum(
            agent_order_stay[order_id][t] *
            upgrade_amount[order_id, low_type, room_type]
            for order_id in agent_order_set
            for low_type in
                [i for i in room_type_set if room_type in upgrade_levels[i]]
        )
        >= effective_sale_for_individual[room_type, t, outcome_id]
        for room_type, t, outcome_id in demand_indice),
        name="Individual sales limited by the vacancy"
    )

    model.addConstrs(
        (
            individual_demand_pmf[room_type][t][outcome_id]['quantity'] >=
            effective_sale_for_individual[room_type, t, outcome_id]
            for room_type, t, outcome_id in demand_indice
        ),
        name="Individual sales limited by the realization of demand"
    )

    model.setObjective(
        quicksum(agent_order_price[order_id] * order_acceptance[order_id]
                for order_id in agent_order_set) +
        quicksum(
            agent_order_stay[order_id][t] * quicksum(
                upgrade_fee[room_type][up_type] *
                upgrade_amount[order_id, room_type, up_type]
                for room_type in room_type_set
                for up_type in upgrade_levels[room_type]
            )
            for order_id in agent_order_set for t in time_span
        ) +
        quicksum(
            individual_room_price[room_type] *
            quicksum(
                individual_demand_pmf[room_type][t][str(outcome_id)]['prob'] *
                effective_sale_for_individual[room_type, t, str(outcome_id)]
                for t in time_span
                for outcome_id in range(1, demand_ub[room_type] + 2)
            ) 
            for room_type in room_type_set
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

    # def acc_verbose(order_acceptance):
    #     if order_acceptance == 1:
    #         return 'Accept'
    #     else:
    #         return 'Reject'

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
        names=["room", "time", "outcome"]
    )
    sale = sale.reindex(mul_index)
    sale.columns = ["sale"]
    acc_result = np.array(
        [order_acceptance[str(i + 1)].x for i in range(len(agent_order_set))]
    )
    return acc_result, up_result, model.objVal, sale
    model.close()