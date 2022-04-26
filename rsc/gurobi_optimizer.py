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
    if (type(ind_cancel_val) == int) & (type(ind_reservation_id) == str):
        return binom.pmf(
            ind_cancel_val, 
            id_to_val(ind_reservation_id), 
            cancel_rate
        )
    else:
        raise Exception("wrong type")

class GurobiManager:
    def __init__(self, data_reader, instance_id, upgrade_rule, 
                 with_capacity_reservation=False, with_ind_cancel=False, 
                 with_agent_cancel=False):
        """Create gurobi optimizer object.

        Args:
            data_reader (JSONDataReader): _description_
            instance_id (int): _description_
            upgrade_rule (str): Options are `up`, `down`, and `both`
            with_capacity_reservation (bool, optional): Control model. 
            Defaults to False.
            with_ind_cancel (bool, optional): Control model. Defaults to False.
            with_agent_cancel (bool, optional): Control model. Defaults to False.
        """
        (
            self.agent_order_set, 
            self.time_span, 
            self.agent_order_price, 
            self.agent_order_room_quantity,
            self.agent_order_stay, 
            self.agent_cancel_rate
        ) = data_reader.collect_agent_info(instance_id)
        (
            self.room_type_set, 
            self.upgrade_levels, 
            self.downgrade_levels, 
            self.room_capacity,
            self.upgrade_fee, 
            self.compensation_price
        ) = data_reader.collect_hotel_info(upgrade_rule)
        (
            self.individual_demand_pmf, 
            self.individual_room_price, 
            self.demand_ub,
            self.individual_cancel_rate
        ) = data_reader.collect_individual_info()
        self.with_capacity_reservation = with_capacity_reservation
        self.with_ind_cancel = with_ind_cancel
        self.with_agent_cancel = with_agent_cancel

    def _get_df(self, variable, col_name:str, index_names:list):
        var_sol = self.model.getAttr('x', variable)
        var_df = pd.DataFrame(
            {col_name: var_sol.values()}, 
            index=pd.MultiIndex.from_tuples(var_sol.keys(), names=index_names)
        )
        # var_df = pd.DataFrame.from_dict(
        #     self.model.getAttr('x', variable),
        #     orient="index"
        # )
        # mul_index = pd.MultiIndex.from_tuples(
        #     variable.keys(),
        #     names=index_names
        # )
        # var_df = var_df.reindex(mul_index)
        # var_df.columns = [col_name]
        return var_df

    def _load_basic_indice(self):
        # TODO maybe use itertools would be more clean
        self.upgrade_indice = [
            (k, i, j)
            for k in self.agent_order_set
            for i in self.room_type_set
            for j in self.upgrade_levels[i]
        ]
        # FIXME demand quantity and indice relationship needs consistency
        self.ind_demand_indice = [
            (room_type, t, demand_id)
            for room_type in self.room_type_set
            for t in self.time_span
            for demand_id in 
            (np.arange((self.demand_ub[room_type] + 1)) + 1).astype(str)
        ]
    def _load_complement_indice(self):
        if self.with_ind_cancel:
            self.ind_demand_reservation_indice = [
                (room_type, t, demand_id, reservation_id)
                for room_type in self.room_type_set
                for t in self.time_span
                for demand_id in 
                (np.arange((self.demand_ub[room_type] + 1)) + 1).astype(str)
                for reservation_id in 
                (np.arange(int(demand_id)) + 1).astype(str)
            ]
        if self.with_capacity_reservation:
            self.room_time_indice = list(
                product(self.room_type_set, self.time_span)
            )
        
    def _load_indice(self):
        self._load_basic_indice()
        self._load_complement_indice()

    def _create_basic_variable(self):
        self.order_acceptance = self.model.addVars(
            self.agent_order_set, 
            vtype=GRB.BINARY,
            name=f'Order acceptance'
        )
        self.upgrade_amount = self.model.addVars(
            self.upgrade_indice, 
            vtype=GRB.CONTINUOUS,
            name=f'Upgrade amount'
        )
        self.individual_reservation = self.model.addVars(
            self.ind_demand_indice,
            vtype=GRB.CONTINUOUS,
            name=f'Sale for individuals'
        )

    def _create_complement_variable(self):
        if self.with_ind_cancel:
            self.is_valid_reservation = self.model.addVars(
                self.ind_demand_reservation_indice, 
                vtype=GRB.BINARY,  
                name='If indice equals the reservation amount'
            )

    def _create_variable(self):
        self._create_basic_variable()
        self._create_complement_variable()

    def _add_constraints(self):
        
        ## basic constraint
        # no indirect upgrades
        self.model.addConstrs(
            (
                quicksum(self.upgrade_amount[order_id, room_type, up_type]
                         for up_type in self.upgrade_levels[room_type]) <=
                (self.agent_order_room_quantity[order_id][room_type] *
                 self.order_acceptance[order_id])
                for order_id in self.agent_order_set
                for room_type in self.room_type_set
            ),
            name="Direct upgrades"
        )
        # limit individual reservation by demand
        self.model.addConstrs(
            (
                self.individual_reservation[
                    room_type, t, demand_id
                ] <= id_to_val(demand_id)
                for room_type, t, demand_id in self.ind_demand_indice
            ),
            name="individual reservation UB by demand"
        )

        ## complement constraints
        if not self.with_capacity_reservation:
            # agent consumption can not exceed capacity
            self.model.addConstrs(
                (
                    quicksum(
                        self.agent_order_stay[order_id][t] * 
                        self.agent_order_room_quantity[order_id][room_type] *
                        self.order_acceptance[order_id]
                        for order_id in self.agent_order_set
                    )
                    - quicksum(
                        self.agent_order_stay[order_id][t] * 
                        self.upgrade_amount[order_id, room_type, up_type] 
                        for order_id in self.agent_order_set
                        for up_type in self.upgrade_levels[room_type] 
                    )
                    + quicksum(
                        self.agent_order_stay[order_id][t] * 
                        self.upgrade_amount[order_id, original_type, room_type]
                        for order_id in self.agent_order_set
                        for original_type in self.downgrade_levels[room_type] 
                    ) <= 
                    self.room_capacity[room_type]
                    for room_type in self.room_type_set
                    for t in self.time_span
                ),
                name='Capacity UB on agent orders'
            )
            # limit individual reservation by left vacancy
            self.model.addConstrs(
                (
                    self.individual_reservation[
                        room_type, t, demand_id
                    ] <= 
                    self.room_capacity[room_type] 
                    - quicksum(
                        self.agent_order_stay[order_id][t] * 
                        self.agent_order_room_quantity[order_id][room_type] *
                        self.order_acceptance[order_id]
                        for order_id in self.agent_order_set
                    )
                    + quicksum(
                        self.agent_order_stay[order_id][t] * 
                        self.upgrade_amount[order_id, room_type, up_type] 
                        for order_id in self.agent_order_set
                        for up_type in self.upgrade_levels[room_type] 
                    )
                    - quicksum(
                        self.agent_order_stay[order_id][t] * 
                        self.upgrade_amount[order_id, original_type, room_type]
                        for order_id in self.agent_order_set
                        for original_type in self.downgrade_levels[room_type] 
                    )
                    for room_type, t, demand_id in self.ind_demand_indice
                ),
                name="Individual reservation UB by left vacancy"
            )

        if self.with_ind_cancel:
            # identity which reservation id is valid and realized
            self.model.addConstrs(
                (
                    (1 - self.is_valid_reservation[room_type, t, demand_id, 
                                                   reservation_id])
                    * id_to_val(demand_id) >=
                    id_to_val(reservation_id) 
                    - self.individual_reservation[room_type, t, demand_id]
                    for room_type, t, demand_id, reservation_id in 
                    self.ind_demand_reservation_indice
                ),
                name='reservation identification by overflow filter'
            )
            self.model.addConstrs(
                (
                    (self.is_valid_reservation[room_type, t, demand_id, 
                                               reservation_id] 
                     - 1)
                    * id_to_val(demand_id) <=
                    id_to_val(reservation_id) 
                    - self.individual_reservation[room_type, t, demand_id]
                    for room_type, t, demand_id, reservation_id in 
                    self.ind_demand_reservation_indice
                ),
                name='reservation identification by underflow filter'
            )
            

    def _set_objective_func(self):
        
        if (self.with_ind_cancel & (not self.with_agent_cancel) & 
            (not self.with_capacity_reservation)):
            self.model.setObjective(
                quicksum(
                    self.agent_order_price[order_id] 
                    * self.order_acceptance[order_id]
                    for order_id in self.agent_order_set
                ) +
                quicksum(
                    self.agent_order_stay[order_id][t] * (
                        quicksum(
                            self.upgrade_fee[room_type][up_type] * 
                            self.upgrade_amount[order_id, room_type, up_type]
                            for room_type in self.room_type_set
                            for up_type in self.upgrade_levels[room_type]
                        )
                    )
                    for order_id in self.agent_order_set
                    for t in self.time_span
                ) +
                quicksum(
                    self.individual_demand_pmf[room_type][t][demand_id]['prob'] 
                    * self.is_valid_reservation[room_type, t, demand_id, 
                                                reservation_id]
                    * quicksum(
                        get_ind_cancel_prob(
                            self.individual_cancel_rate[room_type],
                            reservation_id,
                            cancel_val
                        )
                        * (id_to_val(reservation_id) - cancel_val)
                        * self.individual_room_price[room_type]
                        for cancel_val in range(int(reservation_id))
                    )
                    for room_type, t, demand_id, reservation_id in 
                    self.ind_demand_reservation_indice
                ),
                GRB.MAXIMIZE
            )



    def build_model(self,):
        # load indice first
        self._load_indice()
        # build model
        self.model = Model("hotel_booking")
        # create variables
        self._create_variable()
        # add constraints
        self._add_constraints()
        # set objective function
        self._set_objective_func()


    def solve(self, time_limit, mip_gap):
        self.model.Params.TimeLimit = time_limit
        self.model.Params.MIPGap = mip_gap
        self.model.optimize()

    def get_result(self):
        """Return solutions and objective function value in df format

        Returns:
            acc_df (df),
            upgrade_df (df),
            ind_valid_df (df),
            objVal (float)
        """
        upgrade_df = self._get_df(self.upgrade_amount, 'upgrade amount', 
                               ['order', 'from type', 'to type'])
        ind_valid_df = self._get_df(
            self.is_valid_reservation, 
            'valid',
            ['room type', 'time', 'demand ID', 'reservation ID']
        )
        acc_df = self._get_df(self.order_acceptance, 'accept', ['order ID'])
        return acc_df, upgrade_df, ind_valid_df, self.model.objVal

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
    