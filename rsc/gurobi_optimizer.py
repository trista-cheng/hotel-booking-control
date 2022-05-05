import pandas as pd
import numpy as np

from gurobipy import Model, GRB, quicksum, multidict, LinExpr
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
    if (type(ind_cancel_val) == int) & (type(ind_reservation_id) != int):
        return binom.pmf(
            ind_cancel_val,
            id_to_val(ind_reservation_id),
            cancel_rate
        )
    else:
        raise Exception("wrong type")

# use matrix product to calculate prduct probability
"""
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
"""
# multiply one by one with order index
def get_agent_cancel_prob(agent_cancel_outcome: dict, agent_cancel_prob: dict):
    prob = 1
    for order_id in agent_cancel_prob:
        if agent_cancel_outcome[order_id] == 1:
            prob = prob * agent_cancel_prob[order_id]
        else:
            prob = prob * (1 - agent_cancel_prob[order_id])
    return prob

# outer product indice together
"""
by pandas
chamber_product_time_index = pd.DataFrame(chamber_product_index).merge(
    pd.DataFrame(time_set.reshape((-1, 1)), columns=['time']),
    how='cross'
)
chamber_product_time_index.to_dict('split')['data']
list(chamber_product_time_index.itertuples(index=False, name=None))
"""
# by numpy
def product_index(origin_index, added_index):
    """combine 1 or N-D index with another 1D array

    Args:
        origin_index (list): 1 or N-D
        added_index (list): 1-D

    Returns:
        tuple of outer join index
    """
    num_product_entry = len(origin_index) * len(added_index)
    extended_origin = np.repeat(
        origin_index, repeats=len(added_index), axis=0)
    if len(extended_origin.shape) == 1:
        extended_origin = np.reshape(extended_origin, (-1, 1))
    index = np.hstack([
        extended_origin,
        np.resize(np.reshape(added_index, (-1, 1)), (num_product_entry, 1))
    ])
    return tuple(map(tuple, index))

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
        if len(index_names) == 1:
            index = None
        else:
            index = pd.MultiIndex.from_tuples(var_sol.keys(), names=index_names)
            
        var_df = pd.DataFrame(
            {col_name: var_sol.values()},
            index=index
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

        if self.with_agent_cancel:
            agent_cancel_df = pd.DataFrame(
                list(product([0, 1], repeat=len(self.agent_order_set))),
                columns=self.agent_order_set
            )
            agent_cancel_df.index = (agent_cancel_df.index + 1).astype(str)
            self.agent_cancel_indice = agent_cancel_df.index.to_list()
            # TODO nested key: cancel_id x agent_order_id
            self.agent_cancel_comb = agent_cancel_df.to_dict(orient='index')
            # self.agent_cancel_indice, self.agent_cancel_comb = multidict(
            #     agent_cancel_df.T.to_dict(orient='list')
            # )
            self.agent_cancel_order_indice = list(
                product(self.agent_cancel_indice, self.agent_order_set)
            )

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

        # FIXME agent cancellation exists only if `with_ind_cancel` is True
        if (self.with_ind_cancel & self.with_capacity_reservation):
            # not used for further model
            ind_realization_indice = [
                (room_type, t, demand_id, reservation_id, ind_cancel_id)
                for room_type in self.room_type_set
                for t in self.time_span
                for demand_id in
                (np.arange((self.demand_ub[room_type] + 1)) + 1).astype(str)
                for reservation_id in
                (np.arange(int(demand_id)) + 1).astype(str)
                for ind_cancel_id in
                (np.arange(int(reservation_id)) + 1).astype(str)
            ]
            if self.with_agent_cancel:
                self.realization_indice = product_index(
                    ind_realization_indice,
                    self.agent_cancel_indice
                )
            else:
                self.realization_indice = ind_realization_indice

        # if self.with_capacity_reservation:
        #     self.room_time_indice = list(
        #         product(self.room_type_set, self.time_span)
        #     )

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

        if self.with_agent_cancel:
            self.agent_order_realization = self.model.addVars(
                self.agent_cancel_order_indice,
                vtype=GRB.BINARY,
                name='Realization of agent orders'
            )

        if self.with_capacity_reservation:
            self.capacity_reservation = self.model.addVars(
                self.room_type_set,
                self.time_span,
                vtype=GRB.CONTINUOUS,
                name='Capacity reserved for individual'
            )

        # TODO It covers if `with_agent_cancel` is True and False
        if (self.with_ind_cancel & self.with_capacity_reservation):
            self.compensation_room_amount = self.model.addVars(
                self.realization_indice,
                lb=0,
                vtype=GRB.CONTINUOUS,
                name='Number of rooms to compensate'
            )

    def _create_variable(self):
        self._create_basic_variable()
        self._create_complement_variable()

    def _add_constraints(self):

        ## basic constraint
        # no indirect upgrades
        for order_id in self.agent_order_set:
            for room_type in self.room_type_set:
                self.model.addConstr(
                    quicksum(
                        self.upgrade_amount[order_id, room_type, up_type]
                        for up_type in self.upgrade_levels[room_type]
                    ) <= self.agent_order_room_quantity[order_id][room_type] *
                    self.order_acceptance[order_id],
                    name="Direct upgrades"
                )
        # limit individual reservation by demand
        for room_type, t, demand_id in self.ind_demand_indice:
            self.model.addConstr(
                (
                    self.individual_reservation[
                        room_type, t, demand_id
                    ] <= id_to_val(demand_id)
                ),
                name="individual reservation UB by demand"
            )

        ## complement constraints
        if not self.with_capacity_reservation:
            
            # agent consumption can not exceed capacity
            # det_agent_consump = dict()
            for room_type in self.room_type_set:
                type_capacity = self.room_capacity[room_type]
                for t in self.time_span:

                    consump = LinExpr(0)
                    for order_id in self.agent_order_set:
                        # TODO compare creating obj or LinExpr obj
                        order_stay = self.agent_order_stay[order_id][t]
                        consump.add(
                            order_stay *
                            self.agent_order_room_quantity[order_id][room_type] *
                            self.order_acceptance[order_id]
                        )
                        # TODO compare quicksum, addTerms and add or divide to 
                        # two parts since order_stay is static
                        for to_type in self.upgrade_levels[room_type]:
                            consump.add(
                                - order_stay * 
                                self.upgrade_amount[order_id, room_type, to_type]
                            )
                        # consump.add(- order_stay *
                        #     quicksum(
                        #         self.upgrade_amount[order_id, room_type, to_type]
                        #         for to_type in self.upgrade_levels[room_type]
                        #     )
                        # )
                        for from_type in self.downgrade_levels[room_type]:
                            consump.add(
                                order_stay *
                                self.upgrade_amount[order_id, from_type, room_type]
                            )

                    self.model.addConstr(
                        consump <= type_capacity,
                        name='Capacity UB on agent orders'
                    )
                    # det_agent_consump[(room_type, t)] = consump
                    for demand_id in self.demand_ub[room_type]:
                        self.model.addConstr(
                            self.individual_reservation[
                                room_type, t, demand_id
                            ] <= type_capacity - consump,
                            name="Individual reservation UB by left vacancy"
                        )

            # limit individual reservation by left vacancy
            # for room_type, t, demand_id in self.ind_demand_indice:
            #     self.model.addConstr(
            #         self.individual_reservation[
            #             room_type, t, demand_id
            #         ] <=
            #         self.room_capacity[room_type] - det_agent_consump[room_type, t],
            #         name="Individual reservation UB by left vacancy"
            #     )

        else:  # with capacity reservation
            for room_type, t, demand_id in self.ind_demand_indice:
                self.model.addConstr(
                    (
                        self.individual_reservation[
                            room_type, t, demand_id
                        ] <= self.capacity_reservation[room_type, t]
                    ),
                    name="individual reservation UB by conserved capacity"
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

        if self.with_agent_cancel:
            # define agent order realization
            self.model.addConstrs(
                (
                    self.agent_order_realization[agent_cancel_id, order_id] <=
                    self.order_acceptance[order_id]
                    for agent_cancel_id, order_id in
                    self.agent_cancel_order_indice
                ),
                name='order realization limited by acceptance'
            )
            self.model.addConstrs(
                (
                    self.agent_order_realization[agent_cancel_id, order_id] <=
                    1 - self.agent_cancel_comb[agent_cancel_id][order_id]
                    for agent_cancel_id, order_id in
                    self.agent_cancel_order_indice
                ),
                name='order realization limited by cancellation'
            )

        if (self.with_agent_cancel & self.with_ind_cancel &
            self.with_capacity_reservation):
            self.model.addConstrs(
                (
                    self.compensation_room_amount[room_type, t, ind_demand_id,
                                                  ind_reservation_id,
                                                  ind_cancel_id,
                                                  agent_cancel_id] >=
                    - self.room_capacity[room_type]
                    + id_to_val(ind_reservation_id)
                    - id_to_val(ind_cancel_id)
                    + quicksum(
                        self.agent_order_stay[order_id][t] *
                        self.agent_order_room_quantity[order_id][room_type] *
                        self.agent_order_realization[agent_cancel_id, order_id]
                        for order_id in self.agent_order_set
                    )
                    - quicksum(
                        self.agent_order_stay[order_id][t] *
                        self.upgrade_amount[order_id, room_type, up_type] *
                        (1 - self.agent_cancel_comb[agent_cancel_id][order_id])
                        for order_id in self.agent_order_set
                        for up_type in self.upgrade_levels[room_type]
                    )
                    + quicksum(
                        self.agent_order_stay[order_id][t] *
                        self.upgrade_amount[order_id, original_type, room_type]
                        * (1 -
                           self.agent_cancel_comb[agent_cancel_id][order_id])
                        for order_id in self.agent_order_set
                        for original_type in self.downgrade_levels[room_type]
                    )
                    - (1 - self.is_valid_reservation[room_type, t,
                                                     ind_demand_id,
                                                     ind_reservation_id])
                    * (
                        quicksum(
                            self.room_capacity[iter_room_id]
                            for iter_room_id in self.room_type_set
                        )
                        + id_to_val(ind_reservation_id)
                    )
                    for (room_type, t, ind_demand_id, ind_reservation_id,
                         ind_cancel_id, agent_cancel_id) in
                    self.realization_indice
                ),
                name='compensation room amount LB'
            )

        if ((not self.with_agent_cancel) & self.with_ind_cancel &
            self.with_capacity_reservation):
            self.model.addConstrs(
                (
                    self.compensation_room_amount[room_type, t, ind_demand_id,
                                                  ind_reservation_id,
                                                  ind_cancel_id] >=
                    - self.room_capacity[room_type]
                    + id_to_val(ind_reservation_id)
                    - id_to_val(ind_cancel_id)
                    + quicksum(
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
                    )
                    - (1 - self.is_valid_reservation[room_type, t,
                                                     ind_demand_id,
                                                     ind_reservation_id])
                    * (
                        quicksum(
                            self.room_capacity[iter_room_id]
                            for iter_room_id in self.room_type_set
                        )
                        + id_to_val(ind_reservation_id)
                    )
                    for (room_type, t, ind_demand_id, ind_reservation_id,
                         ind_cancel_id) in
                    self.realization_indice
                ),
                name='compensation room amount LB'
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

        if (self.with_ind_cancel & self.with_agent_cancel &
            (not self.with_capacity_reservation)):
            self.model.setObjective(
                quicksum(
                    get_agent_cancel_prob(
                        self.agent_cancel_comb[agent_cancel_id],
                        self.agent_cancel_rate
                    )
                    * (
                        self.agent_order_price[order_id] *
                        self.agent_order_realization[agent_cancel_id, order_id]
                        + quicksum(
                            self.agent_order_stay[order_id][t] * (
                                quicksum(
                                    self.upgrade_fee[room_type][up_type] *
                                    self.upgrade_amount[order_id, room_type, 
                                                        up_type] *
                                    (1 - self.agent_cancel_comb[
                                        agent_cancel_id][order_id])
                                    for room_type in self.room_type_set
                                    for up_type in 
                                    self.upgrade_levels[room_type]
                                )
                            )
                            for t in self.time_span
                        )
                    )
                    for agent_cancel_id, order_id in 
                    self.agent_cancel_order_indice
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

        if (self.with_ind_cancel & (not self.with_agent_cancel) &
            self.with_capacity_reservation):
            self.model.setObjective(
                quicksum(
                    self.agent_order_price[order_id] *
                    self.order_acceptance[order_id]
                    + quicksum(
                        self.agent_order_stay[order_id][t] * (
                            quicksum(
                                self.upgrade_fee[room_type][up_type] *
                                self.upgrade_amount[order_id, room_type, 
                                                    up_type] 
                                for room_type in self.room_type_set
                                for up_type in 
                                self.upgrade_levels[room_type]
                            )
                        )
                        for t in self.time_span
                    )
                    for order_id in self.agent_order_set
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
                ) -
                quicksum(
                    self.individual_demand_pmf[room_type][t][ind_demand_id]
                                              ['prob']
                    * get_ind_cancel_prob(
                        self.individual_cancel_rate[room_type],
                        ind_reservation_id,
                        id_to_val(ind_cancel_id)
                    )
                    * self.compensation_price[room_type][t] 
                    * self.compensation_room_amount[room_type, t, ind_demand_id, 
                                                    ind_reservation_id, 
                                                    ind_cancel_id]
                    for (room_type, t, ind_demand_id, ind_reservation_id, 
                         ind_cancel_id) 
                    in self.realization_indice
                ),
                GRB.MAXIMIZE
            )

        if (self.with_ind_cancel & self.with_agent_cancel &
            self.with_capacity_reservation):
            self.model.setObjective(
                quicksum(
                    get_agent_cancel_prob(
                        self.agent_cancel_comb[agent_cancel_id],
                        self.agent_cancel_rate
                    )
                    * (
                        self.agent_order_price[order_id] *
                        self.agent_order_realization[agent_cancel_id, order_id]
                        + quicksum(
                            self.agent_order_stay[order_id][t] * (
                                quicksum(
                                    self.upgrade_fee[room_type][up_type] *
                                    self.upgrade_amount[order_id, room_type, 
                                                        up_type] *
                                    (1 - self.agent_cancel_comb[
                                        agent_cancel_id][order_id])
                                    for room_type in self.room_type_set
                                    for up_type in 
                                    self.upgrade_levels[room_type]
                                )
                            )
                            for t in self.time_span
                        )
                    )
                    for agent_cancel_id, order_id in 
                    self.agent_cancel_order_indice
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
                ) -
                quicksum(
                    self.individual_demand_pmf[room_type][t][ind_demand_id]
                                              ['prob']
                    * get_ind_cancel_prob(
                        self.individual_cancel_rate[room_type],
                        ind_reservation_id,
                        id_to_val(ind_cancel_id)
                    )
                    * get_agent_cancel_prob(
                        self.agent_cancel_comb[agent_cancel_id],
                        self.agent_cancel_rate
                    )
                    * self.compensation_price[room_type][t] 
                    * self.compensation_room_amount[room_type, t, ind_demand_id, 
                                                    ind_reservation_id, 
                                                    ind_cancel_id, 
                                                    agent_cancel_id]
                    for (room_type, t, ind_demand_id, ind_reservation_id, 
                         ind_cancel_id, agent_cancel_id) 
                    in self.realization_indice
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
        if self.with_ind_cancel:
            ind_valid_df = self._get_df(
                self.is_valid_reservation,
                'valid',
                ['room type', 'time', 'demand ID', 'reservation ID']
            )
        else:
            ind_valid_df = pd.DataFrame()
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
