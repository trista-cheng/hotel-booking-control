import time
import numpy as np

from gurobipy import Model, GRB, quicksum

from data_reader import JSONDataReader
from data_manager import CAPACITY, INDIVIDUAL_POP_SIZE

def solve(data_reader, instance_id):
    (agent_order_set, time_span, agent_order_price, agent_order_room_quantity,
        agent_order_stay) = data_reader.collect_agent_info(instance_id)
    room_type_set, upgrade_levels, downgrade_levels, room_capacity, upgrade_fee = \
        data_reader.collect_hotel_info(instance_id)
    individual_demand_pmf, individual_room_price = \
        data_reader.collect_individual_info(instance_id)

    upgrade_indice = [
        (k, i, j)
        for k in agent_order_set
        for i in agent_order_room_quantity[k]
        for j in upgrade_levels[i]
    ]
    demand_scenario_indice = [
        (room_type, t, str(demand_id))
        for room_id, room_type in enumerate(room_type_set)
        for t in time_span
        for demand_id in range(1, np.min([INDIVIDUAL_POP_SIZE[room_id], room_capacity[room_type]]) + 2)
    ]

    model = Model("hotel_booking")
    order_acceptance = model.addVars(agent_order_set, vtype=GRB.BINARY,
                                    name=f'Order acceptance')
    upgrade_amount = model.addVars(upgrade_indice, vtype=GRB.INTEGER,
                                name=f'upgrade amount')
    effective_sale_for_individual = model.addVars(demand_scenario_indice,
                                                vtype=GRB.INTEGER,
                                                name=f'Sale for individuals')

    # no indirect upgrades
    model.addConstrs(
        quicksum(upgrade_amount[order_id, room_type, up_type]
                for up_type in upgrade_levels[room_type]) <=
        agent_order_room_quantity[order_id][room_type] * order_acceptance[order_id]
        for order_id in agent_order_set
        for room_type in agent_order_room_quantity[order_id]
    )

    # capacity constraint
    model.addConstrs(
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
    )

    # define the effective sale for individual
    model.addConstrs(
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
        >= effective_sale_for_individual[room_type, t, scenario_id]
        for room_type, t, scenario_id in demand_scenario_indice
    )

    model.addConstrs(
        individual_demand_pmf[room_type][t][scenario_id]['quantity'] >=
        effective_sale_for_individual[room_type, t, scenario_id]
        for room_type, t, scenario_id in demand_scenario_indice
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
                individual_demand_pmf[room_type][t][scenario_id]['prob'] *
                effective_sale_for_individual[room_type, t, scenario_id]
                for t in time_span
                for scenario_id in individual_demand_pmf[room_type][t]
            )
            for room_type in room_type_set
        ),
        GRB.MAXIMIZE
    )
    # model.Params.TimeLimit = float('inf')
    model.Params.TimeLimit = 20
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
    #     for scenario_id in individual_demand_pmf[room_type]:
    #         print(scenario_id, ':', individual_demand_pmf[room_type][scenario_id],)
    #         for t in time_span:
    #             print(effective_sale_for_individual[room_type, t, scenario_id].x, end=', ')
    #         print()
    #     print('='*20)

    # model.write('mip1.rlp')

    up_result = np.zeros((len(agent_order_set), len(room_type_set), len(room_type_set)))
    for indice in upgrade_indice:
        up_result[int(indice[0]) -1, int(indice[1]) -1, int(indice[2]) -1] = upgrade_amount[indice].x
    return np.array([order_acceptance[i].x for i in agent_order_set]), up_result, agent_order_room_quantity, agent_order_stay, room_capacity, model.objVal
    model.close()