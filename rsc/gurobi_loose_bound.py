import pandas as pd
import numpy as np

from gurobipy import Model, GRB, quicksum, multidict, LinExpr
from itertools import product
from scipy.stats import binom

from order_replier import OrderManager
from data_reader import JSONDataReader

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
def get_agent_cancel_prob(agent_cancel_outcome: dict, agent_cancel_prob: dict,
                          agent_order_set):
    prob = 1
    for order_id in agent_order_set:
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

class GurobiBoundary:
    def __init__(self, scenario, instance_id, upgrade_rule,
                 with_capacity_reservation, with_ind_cancel,
                 with_agent_cancel, set_order_acc=False):
        """Create gurobi optimizer object.

        Args:
            scenario (dict): parameter for DataReader
            instance_id (int): _description_
            upgrade_rule (str): Options are `up`, `down`, and `both`
            with_capacity_reservation (bool): Only allowed to be True.
            with_ind_cancel (bool): Control model. Defaults to False.
            with_agent_cancel (bool): Control model. Defaults to False.
            set_order_acc (bool): Must be False
        """
        if not with_capacity_reservation:
            raise Exception("Loose bound is only retrieved in complicated envs")
        if set_order_acc:
            raise Exception("Loose boundary is not for algorithm")
        data_reader = JSONDataReader(scenario)
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
        self.set_order_acc = set_order_acc


    def _get_df(self, variable, col_name:str, index_names:list):
        var_sol = self.model.getAttr('x', variable)
        if len(index_names) == 1:
            index = var_sol.keys()
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
            for demand_id in \
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
                for demand_id in \
                (np.arange((self.demand_ub[room_type] + 1)) + 1).astype(str)
                for reservation_id in \
                (np.arange(int(demand_id)) + 1).astype(str)
            ]

        # RELEASE and REMOVE
        # FIXME agent cancellation exists only if `with_ind_cancel` is True
        # if (self.with_ind_cancel & self.with_capacity_reservation):
        #     # not used for further model
        #     ind_realization_indice = [
        #         (room_type, t, demand_id, reservation_id, ind_cancel_id)
        #         for room_type in self.room_type_set
        #         for t in self.time_span
        #         for demand_id in \
        #         (np.arange((self.demand_ub[room_type] + 1)) + 1).astype(str)
        #         for reservation_id in \
        #         (np.arange(int(demand_id)) + 1).astype(str)
        #         for ind_cancel_id in \
        #         (np.arange(int(reservation_id)) + 1).astype(str)
        #     ]
        #     if self.with_agent_cancel:
        #         self.realization_indice = product_index(
        #             ind_realization_indice,
        #             self.agent_cancel_indice
        #         )
        #     else:
        #         self.realization_indice = ind_realization_indice

        # if self.with_capacity_reservation:
        #     self.room_time_indice = list(
        #         product(self.room_type_set, self.time_span)
        #     )

    def _load_indice(self):
        self._load_basic_indice()
        self._load_complement_indice()

    def _create_basic_variable(self):
        if not self.set_order_acc:
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
            lb=0,
            name=f'Sale for individuals'
        )

    def _create_complement_variable(self):

        if self.with_ind_cancel:
            self.is_valid_reservation = self.model.addVars(
                self.ind_demand_reservation_indice,
                vtype=GRB.BINARY,
                name='If indice equals the reservation amount'
            )
            self.if_individual_underage = self.model.addVars(
                self.ind_demand_indice,
                vtype=GRB.BINARY,
                name=('If demand exceeds the amount for individuals and'
                      ' there is unsatisfied demand')
            )

        if self.with_agent_cancel:
            self.agent_order_realization = self.model.addVars(
                self.agent_cancel_order_indice,
                vtype=GRB.BINARY,
                name='Realization of agent orders'
            )

        if self.with_capacity_reservation:
            # set UB
            agent_usage_ub = {}
            for t in self.time_span:
                agent_usage_ub[t] = quicksum(
                    self.agent_order_room_quantity[iter_order_id]
                                                  [iter_room_type] *
                    self.agent_order_stay[iter_order_id][t]
                    for iter_order_id in self.agent_order_set
                    for iter_room_type in self.room_type_set
                )
            self.agent_usage_ub = agent_usage_ub
            self.capacity_reservation = self.model.addVars(
                self.room_type_set,
                self.time_span,
                vtype=GRB.CONTINUOUS,
                name='Capacity reserved for individual'
            )

        # RELEASE and REMOVE
        # # TODO It covers if `with_agent_cancel` is True and False
        # if (self.with_ind_cancel & self.with_capacity_reservation):
        #     self.compensation_room_amount = self.model.addVars(
        #         self.realization_indice,
        #         lb=0,
        #         vtype=GRB.CONTINUOUS,
        #         name='Number of rooms to compensate'
        #     )

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
            self.model.addConstr(
                self.individual_reservation[room_type, t, demand_id] >=
                id_to_val(demand_id) -
                self.if_individual_underage[room_type, t, demand_id] *
                id_to_val(demand_id)
            )

        # RELEASE and REMOVE
        ## complement constraints
        # if not self.with_capacity_reservation:

        #     # agent consumption can not exceed capacity
        #     # det_agent_consump = dict()
        #     for room_type in self.room_type_set:
        #         type_capacity = self.room_capacity[room_type]
        #         for t in self.time_span:

        #             det_agent_consump = LinExpr(0)
        #             for order_id in self.agent_order_set:
        #                 # TODO compare creating obj or LinExpr obj
        #                 order_stay = self.agent_order_stay[order_id][t]
        #                 det_agent_consump.add(
        #                     order_stay *
        #                     self.agent_order_room_quantity[order_id][room_type] *
        #                     self.order_acceptance[order_id]
        #                 )
        #                 # TODO compare quicksum, addTerms and add or divide to
        #                 # two parts since order_stay is static
        #                 for to_type in self.upgrade_levels[room_type]:
        #                     det_agent_consump.add(
        #                         - order_stay *
        #                         self.upgrade_amount[order_id, room_type, to_type]
        #                     )
        #                 # det_agent_consump.add(- order_stay *
        #                 #     quicksum(
        #                 #         self.upgrade_amount[order_id, room_type, to_type]
        #                 #         for to_type in self.upgrade_levels[room_type]
        #                 #     )
        #                 # )
        #                 for from_type in self.downgrade_levels[room_type]:
        #                     det_agent_consump.add(
        #                         order_stay *
        #                         self.upgrade_amount[order_id, from_type, room_type]
        #                     )

        #             self.model.addConstr(
        #                 det_agent_consump <= type_capacity,
        #                 name='Capacity UB on agent orders'
        #             )
        #             # det_agent_consump[(room_type, t)] = det_agent_consump
        #             for demand_id in (np.arange(self.demand_ub[room_type] + 1) + 1).astype(str):
        #                 left_vac = type_capacity - det_agent_consump
        #                 self.model.addConstr(
        #                     self.individual_reservation[
        #                         room_type, t, demand_id
        #                     ] <= left_vac,
        #                     name="Individual reservation UB by left vacancy"
        #                 )
        #                 # to control ind rev is exactly correct
        #                 self.model.addConstr(
        #                     self.individual_reservation[
        #                         room_type, t, demand_id
        #                     ] >= left_vac -
        #                     ((1 - self.if_individual_underage[room_type, t,
        #                                                       demand_id]) *
        #                      type_capacity),
        #                     name="Individual reservation UB by left vacancy"
        #                 )
        #                 self.model.addConstr(
        #                     self.if_individual_underage[
        #                         room_type, t, demand_id
        #                     ] * id_to_val(demand_id) >=
        #                     id_to_val(demand_id) - left_vac
        #                 )
        #                 self.model.addConstr(
        #                     (self.if_individual_underage[room_type, t,
        #                                                  demand_id] - 1) *
        #                     type_capacity <=
        #                     id_to_val(demand_id) - left_vac
        #                 )


        if self.with_capacity_reservation:  # with capacity reservation
            capacity_reservation_ub = quicksum(
                self.demand_ub[iter_type] for iter_type in self.room_type_set
            )
            for room_type, t, demand_id in self.ind_demand_indice:
                self.model.addConstr(
                    (
                        self.individual_reservation[
                            room_type, t, demand_id
                        ] <= self.capacity_reservation[room_type, t]
                    ),
                    name="individual reservation UB by conserved capacity"
                )
                # to control ind rev exactly correct
                self.model.addConstr(
                    self.individual_reservation[room_type, t, demand_id] >=
                    self.capacity_reservation[room_type, t] -
                    ((1 - self.if_individual_underage[room_type, t, demand_id])
                     * capacity_reservation_ub)
                )
                self.model.addConstr(
                    self.if_individual_underage[room_type, t, demand_id] *
                    id_to_val(demand_id) >= id_to_val(demand_id) -
                    self.capacity_reservation[room_type, t]
                )
                self.model.addConstr(
                    (self.if_individual_underage[room_type, t, demand_id] - 1) *
                    capacity_reservation_ub <= id_to_val(demand_id) -
                    self.capacity_reservation[room_type, t]
                )

        if self.with_ind_cancel:
            # identity which reservation id is valid and realized
            for room_type, t, demand_id, reservation_id in \
                self.ind_demand_reservation_indice:
                # TODO create an object for variable here may not be good
                reservation_amount = \
                    self.individual_reservation[room_type, t, demand_id]
                self.model.addConstr(
                    (1 - self.is_valid_reservation[room_type, t, demand_id,
                                                   reservation_id]) *
                    id_to_val(demand_id) >= id_to_val(reservation_id)
                    - reservation_amount,
                    name='reservation identification by overflow filter'
                )
                self.model.addConstr(
                    (self.is_valid_reservation[room_type, t, demand_id,
                                               reservation_id] - 1) *
                    id_to_val(demand_id) <= id_to_val(reservation_id) -
                    reservation_amount,
                    name='reservation identification by underflow filter'
                )
            for room_type, t, demand_id in self.ind_demand_indice:
                self.model.addConstr(
                    quicksum(
                        self.is_valid_reservation[room_type, t, demand_id,
                                                  reservation_id]
                        for reservation_id in \
                        (np.arange(int(demand_id)) + 1).astype(str)
                    ) == 1
                )

        if self.with_agent_cancel:
            # define agent order realization
            for agent_cancel_id, order_id in self.agent_cancel_order_indice:
                # TODO order realization deplicate
                self.model.addConstr(
                    self.agent_order_realization[agent_cancel_id, order_id] <=
                    self.order_acceptance[order_id],
                    name='order realization limited by acceptance'
                )
                self.model.addConstr(
                    self.agent_order_realization[agent_cancel_id, order_id] <=
                    1 - self.agent_cancel_comb[agent_cancel_id][order_id],
                    name='order realization limited by cancellation'
                )
                self.model.addConstr(
                    self.agent_order_realization[agent_cancel_id, order_id] >=
                    self.order_acceptance[order_id] -
                    self.agent_cancel_comb[agent_cancel_id][order_id],
                    name='order realization LB by acceptance and cancellation'
                )

        # RELEASE and REMOVE
        # MODIFY
        # if (self.with_agent_cancel & self.with_ind_cancel &
        #     self.with_capacity_reservation):

        #     # TODO consider LinExpr changeVar with capacity constraint
        #     sto_agent_consump = LinExpr(0)
        #     for order_id in self.agent_order_set:
        #         order_stay = self.agent_order_stay[order_id][t]
        #         if_cancel = self.agent_cancel_comb[agent_cancel_id][order_id]
        #         sto_agent_consump.add(
        #             order_stay *
        #             self.agent_order_room_quantity[order_id][room_type] *
        #             self.agent_order_realization[agent_cancel_id, order_id]
        #         )
        #         for to_type in self.upgrade_levels[room_type]:
        #             sto_agent_consump.add(
        #                 - order_stay *
        #                 self.upgrade_amount[order_id, room_type, to_type] *
        #                 (1 - if_cancel)
        #             )
        #         for from_type in self.downgrade_levels[room_type]:
        #             sto_agent_consump.add(
        #                 order_stay *
        #                 self.upgrade_amount[order_id, from_type, room_type]
        #                 * (1 - if_cancel)
        #             )

        #     self.model.addConstr(
        #         sto_agent_consump <= self.room_capacity[room_type]+
        #         id_to_val(ind_reservation_id) -
        #         id_to_val(ind_cancel_id) +
        #         sto_agent_consump -
        #         (1 - self.is_valid_reservation[room_type, t, ind_demand_id,
        #                                         ind_reservation_id]) *
        #         (self.agent_usage_ub[t] + id_to_val(ind_reservation_id)),
        #         name='realized room amount UB'
        #     )

        # if ((not self.with_agent_cancel) & self.with_ind_cancel &
        #     self.with_capacity_reservation):

        #     for room_type, t, ind_demand_id, ind_reservation_id, ind_cancel_id \
        #         in self.realization_indice:

        #         # TODO duplicate with no capacity reservation
        #         det_agent_consump = LinExpr(0)
        #         for order_id in self.agent_order_set:
        #             order_stay = self.agent_order_stay[order_id][t]
        #             det_agent_consump.add(
        #                 order_stay *
        #                 self.agent_order_room_quantity[order_id][room_type] *
        #                 self.order_acceptance[order_id]
        #             )
        #             for to_type in self.upgrade_levels[room_type]:
        #                 det_agent_consump.add(
        #                     - order_stay *
        #                     self.upgrade_amount[order_id, room_type, to_type]
        #                 )
        #             for from_type in self.downgrade_levels[room_type]:
        #                 det_agent_consump.add(
        #                     order_stay *
        #                     self.upgrade_amount[order_id, from_type, room_type]
        #                 )

        #         self.model.addConstr(
        #             self.compensation_room_amount[room_type, t, ind_demand_id,
        #                                           ind_reservation_id,
        #                                           ind_cancel_id] >=
        #             - self.room_capacity[room_type]
        #             + id_to_val(ind_reservation_id)
        #             - id_to_val(ind_cancel_id)
        #             + det_agent_consump
        #             - (1 - self.is_valid_reservation[room_type, t,
        #                                              ind_demand_id,
        #                                              ind_reservation_id])
        #             * (self.agent_usage_ub[t] + id_to_val(ind_reservation_id)),
        #             name='compensation room amount LB'
        #         )


    def _set_objective_func(self):

        # FIXME no objective function for no ind cancel
        # TODO individual profit is duplicate but to reduce the loop, it is
        # combined with compensation

        if not self.with_agent_cancel:
            agent_profit = LinExpr(0)
            for order_id in self.agent_order_set:
                agent_profit.add(
                    self.agent_order_price[order_id]
                    * self.order_acceptance[order_id]
                )
                # for t in self.time_span:
                #     # TODO consider whether to create lInExpr
                #     # upgrade_fee_tonight = LinExpr(0)
                #     agent_profit.add(
                #         self.agent_order_stay[order_id][t] *
                #         quicksum(
                #             self.upgrade_fee[room_type][up_type] *
                #             self.upgrade_amount[order_id, room_type, up_type]
                #             for room_type in self.room_type_set
                #             for up_type in self.upgrade_levels[room_type]
                #         )
                #     )
                # TODO use quicksum instead of for loop as multiplying
                agent_profit.add(
                    quicksum(
                        self.agent_order_stay[order_id][t]
                        for t in self.time_span
                    ) * quicksum(
                        self.upgrade_fee[room_type][up_type] *
                        self.upgrade_amount[order_id, room_type, up_type]
                        for room_type in self.room_type_set
                        for up_type in self.upgrade_levels[room_type]
                    )
                )

        else:
            agent_profit = LinExpr(0)
            for agent_cancel_id, order_id in self.agent_cancel_order_indice:
                agent_profit.add(
                    get_agent_cancel_prob(
                        self.agent_cancel_comb[agent_cancel_id],
                        self.agent_cancel_rate,
                        self.agent_order_set
                    ) * (
                        self.agent_order_price[order_id] *
                        self.agent_order_realization[agent_cancel_id, order_id] +
                        quicksum(
                            self.agent_order_stay[order_id][t]
                            for t in self.time_span
                        ) * quicksum(
                            self.upgrade_fee[room_type][up_type] *
                            self.upgrade_amount[order_id, room_type, up_type] *
                            (1 - self.agent_cancel_comb[agent_cancel_id][order_id])
                            for room_type in self.room_type_set
                            for up_type in self.upgrade_levels[room_type]
                        )
                    )
                )

        if ((not self.with_capacity_reservation) & self.with_ind_cancel):
            # this part actually duplicates in four models, but since the
            # compensation item facilitate almost the same indice, it can not
            # be modulized
            ind_profit = LinExpr(0)
            for room_type, t, demand_id, reservation_id in \
                self.ind_demand_reservation_indice:
                ind_profit.add(
                    self.individual_demand_pmf[room_type][t][demand_id]['prob'] *
                    self.is_valid_reservation[room_type, t, demand_id,
                                              reservation_id] *
                    quicksum(
                        get_ind_cancel_prob(
                            self.individual_cancel_rate[room_type],
                            reservation_id,
                            cancel_val
                        ) *
                        (id_to_val(reservation_id) - cancel_val) *
                        self.individual_room_price[room_type]
                        for cancel_val in range(int(reservation_id))
                    )
                )

        if (self.with_ind_cancel & (not self.with_agent_cancel) &
            (not self.with_capacity_reservation)):
            self.model.setObjective(
                agent_profit + ind_profit,
                GRB.MAXIMIZE
            )

        if (self.with_ind_cancel & self.with_agent_cancel &
            (not self.with_capacity_reservation)):
            self.model.setObjective(
                agent_profit + ind_profit,
                GRB.MAXIMIZE
            )

        if (self.with_ind_cancel & (not self.with_agent_cancel) &
            self.with_capacity_reservation):

            ind_and_comp = LinExpr(0)
            for room_type, t, demand_id, reservation_id in \
                self.ind_demand_reservation_indice:

                ind_demand_prob = self.individual_demand_pmf[room_type][t][demand_id]['prob']

                # TODO try addTerms and add
                # TODO try not to iterate both items together, because the
                # former item does not iterate with variable but only statics
                # TODO test if get_prob consume lots of time
                ind_cancel_prob_list = []
                for ind_cancel_id in (np.arange(int(reservation_id)) + 1).astype(str):

                    ind_cancel_prob = get_ind_cancel_prob(
                        self.individual_cancel_rate[room_type],
                        reservation_id,
                        id_to_val(ind_cancel_id)
                    )
                    ind_cancel_prob_list.append(ind_cancel_prob)

                    # RELEASE and REMOVE
                    # ind_and_comp.add(
                    #     - ind_demand_prob * ind_cancel_prob *
                    #     self.compensation_price[room_type][t] *
                    #     self.compensation_room_amount[room_type, t, demand_id,
                    #                                   reservation_id,
                    #                                   ind_cancel_id]
                    # )
                    # # TODO avoid some parameter query keys too often

                # TODO write with expression or var and coef
                ind_and_comp.add(
                    ind_demand_prob *
                    self.is_valid_reservation[room_type, t, demand_id,
                                              reservation_id] *
                    quicksum(
                        ind_cancel_prob_list[cancel_val]
                        * (id_to_val(reservation_id) - cancel_val)
                        * self.individual_room_price[room_type]
                        for cancel_val in range(int(reservation_id))
                    )
                )

            self.model.setObjective(
                agent_profit + ind_and_comp,
                GRB.MAXIMIZE
            )

        if (self.with_ind_cancel & self.with_agent_cancel &
            self.with_capacity_reservation):

            ind_and_comp = LinExpr(0)
            for room_type, t, demand_id, reservation_id in \
                self.ind_demand_reservation_indice:

                ind_demand_prob = self.individual_demand_pmf[room_type][t][demand_id]['prob']
                ind_cancel_prob_list = []
                for ind_cancel_id in (np.arange(int(reservation_id)) + 1).astype(str):

                    ind_cancel_prob = get_ind_cancel_prob(
                        self.individual_cancel_rate[room_type],
                        reservation_id,
                        id_to_val(ind_cancel_id)
                    )
                    ind_cancel_prob_list.append(ind_cancel_prob)

                    # RELEASE and REMOVE
                    # TODO calculate for agent cancel is time-consuming
                    # for agent_cancel_id in self.agent_cancel_indice:
                    #     ind_and_comp.add(
                    #         - ind_demand_prob * ind_cancel_prob *
                    #         get_agent_cancel_prob(
                    #             self.agent_cancel_comb[agent_cancel_id],
                    #             self.agent_cancel_rate,
                    #             self.agent_order_set
                    #         ) * self.compensation_price[room_type][t] *
                    #         self.compensation_room_amount[
                    #             room_type, t, demand_id, reservation_id,
                    #             ind_cancel_id, agent_cancel_id
                    #         ]
                    #     )

                ind_and_comp.add(
                    ind_demand_prob *
                    self.is_valid_reservation[room_type, t, demand_id,
                                              reservation_id] *
                    quicksum(
                        ind_cancel_prob_list[cancel_val]
                        * (id_to_val(reservation_id) - cancel_val)
                        * self.individual_room_price[room_type]
                        for cancel_val in range(int(reservation_id))
                    )
                )

            self.model.setObjective(
                agent_profit + ind_and_comp,
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
        # REMOVE time and mip gap threshold
        # self.model.Params.TimeLimit = time_limit
        # self.model.Params.MIPGap = mip_gap
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
        if self.set_order_acc:
            acc_df = pd.DataFrame.from_dict(self.order_acceptance, orient='index')
        else:
            acc_df = self._get_df(self.order_acceptance, 'accept', ['order ID', ])
        if self.with_capacity_reservation:
            col = ['room ID', 'time ID', 'demand ID', 'reservation ID',
                   'IND cancel ID']
            if self.with_agent_cancel:
                col += ['agent cancel ID', ]
            comp_df = pd.DataFrame()
            cap_rev_df = self._get_df(self.capacity_reservation, 'cap_rev',
                                      ['room', 'time'])
            cap_rev_df = cap_rev_df.unstack()
            # .reset_index().pivot_table(index='room', columns='time')
        else:
            comp_df = pd.DataFrame()
            cap_rev_df = pd.DataFrame()
        rev_df = self._get_df(self.individual_reservation, 'rev', ['room', 'time', 'demand_ID'])

        return (acc_df, upgrade_df, cap_rev_df, self.model.objVal,
                self.model.MIPGap, ind_valid_df, comp_df, rev_df)

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
