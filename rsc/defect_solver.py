import enum
import numpy as np
import pandas as pd
import time

from data_reader import CSVDataReader
from tools import get_ind_exp_req
from itertools import product
from scipy.stats import binom

BASE_PARAM = 'settings/base.npy'

class Solver:

    def __init__(self, scenario: dict, instance_id: int, upgrade_rule: str,
                 with_capacity_reservation: bool, with_agent_cancel: bool,
                 with_ind_cancel=True, data_root='data'):
        """_summary_

        Args:
            scenario (dict): _description_
            instance_id (int): _description_
            upgrade_rule (str): available strings are `up`, `down`, and `both`
            with_capacity_reservation (bool): _description_
            with_agent_cancel (bool): _description_
            with_ind_cancel (bool, optional): _description_. Defaults to True.
            data_root
        """
        reader = CSVDataReader(scenario=scenario, data_root=data_root)
        (room_type_set, self.room_capacity, self.upgrade_fee,
         self.compensation_price) = \
            reader.collect_hotel_info(upgrade_rule=upgrade_rule)
        (agent_order_set, time_span, self.agent_order_price,
         self.agent_order_room_quantity, self.agent_order_stay,
         self.agent_cancel_rate) = \
            reader.collect_agent_info(instance_id=instance_id)
        (self.individual_demand_pmf, self.individual_room_price, self.demand_ub,
         self.individual_cancel_rate) = reader.collect_individual_info()
        self.time_span_len = len(time_span)
        self.num_order = len(agent_order_set)
        self.num_type = len(room_type_set)
        self.upgrade_rule = upgrade_rule.lower()
        self.scenario = scenario
        self.with_capacity_reservation = with_capacity_reservation
        self.with_agent_cancel = with_agent_cancel
        self.with_ind_cancel = with_ind_cancel
        self.capacity =  (self.room_capacity.reshape((-1, 1)) *
                          np.ones(self.time_span_len))
        if not self.with_ind_cancel:
            self.individual_cancel_rate = np.zeros(self.num_type)
        if not self.with_agent_cancel:
            self.agent_cancel_rate = np.zeros(self.num_order)
        if self.with_capacity_reservation:
            self.cal_agent_cancel_rate = self.agent_cancel_rate.copy()
        else:
            self.cal_agent_cancel_rate = np.zeros(self.num_order)
        with open(BASE_PARAM, 'rb') as f:
            base_setting = np.load(f, allow_pickle=True)[()]
        self.upgrade_fee_gap = base_setting['upgrade_fee_gap_multiplier']

    def get_order_level(self):

        ind_exp_req = get_ind_exp_req(self.scenario, self.time_span_len)
        order_level = self.capacity - ind_exp_req

        return order_level

    def _get_order_rank(self):
        cost = np.dot(
            self.agent_order_room_quantity,
            self.individual_room_price.reshape((-1, 1))
        ).flatten() * self.agent_order_stay.sum(axis=1)
        ratio = (self.agent_order_price / cost)
        if self.with_agent_cancel:
            ratio *= (1 - self.agent_cancel_rate)
        rank = (- ratio).argsort()
        return rank, ratio

    def _get_agent_cancel_table(self, order_acceptance):
        agent_prob_table = []
        agent_real_table = []
        for agent_cancel_outcome in product([0, 1], repeat=self.num_order):
            agent_cancel_outcome = np.array(agent_cancel_outcome)
            mask = np.hstack([agent_cancel_outcome,
                              (1 - agent_cancel_outcome)])
            event_prob_table = np.hstack([
                self.agent_cancel_rate * agent_cancel_outcome,
                (1 - self.agent_cancel_rate) *
                (1 - agent_cancel_outcome)
            ])
            agent_prob = np.prod(event_prob_table,
                                    where=mask.astype(bool))
            agent_real = order_acceptance - agent_cancel_outcome
            agent_real[agent_real < 0] = 0

            agent_prob_table.append(agent_prob)
            agent_real_table.append(agent_real)
        return np.array(agent_prob_table), np.array(agent_real_table)

    def get_obj(self, order_acceptance, order_upgrade, capacity_reservation):

        upgrade_diff = (
            np.dot(order_upgrade, -np.ones((self.num_type, 1))).reshape(
                (self.num_order, self.num_type)
            ) +
            np.dot(np.ones(self.num_type), order_upgrade).reshape(
                (self.num_order, self.num_type)
            )
        )
        acc_order_msk = np.where(order_acceptance == 1)[0]
        # order x room
        acc_order_room = self.agent_order_room_quantity + upgrade_diff
        acc_order_room = acc_order_room[acc_order_msk]
        acc_order_stay = self.agent_order_stay[acc_order_msk]
        # room x time
        agent_det_agg_consump = np.dot(
            acc_order_room.T,
            acc_order_stay
        )

        # validate
        if ((not self.with_capacity_reservation) &
            (agent_det_agg_consump > self.capacity).any()):
            invalid_msk = np.where(agent_det_agg_consump > self.capacity)
            if np.allclose(agent_det_agg_consump[invalid_msk],
                           self.capacity[invalid_msk]):
                pass
            else:
                print(agent_det_agg_consump)
                print(self.capacity)
                raise Exception('Agent consumption exceeds capacity')

        if self.with_agent_cancel:
            # order x room x time
            acc_order_req = acc_order_room[:, :, np.newaxis] * \
                acc_order_stay[: ,np.newaxis, :]
            acc_order_cancel_rate = self.agent_cancel_rate[acc_order_msk]

            # acquiring objects to record order cancel event and probability
            # will use up memory
            # agent_prob_table, agent_real_table = self._get_agent_cancel_table(
            #     order_acceptance
            # )

        # 1D order
        acc_order_upgrade_profit = (
            order_upgrade * self.upgrade_fee *
            self.agent_order_stay.sum(axis=1).reshape(self.num_order, 1, 1)
        ).sum(axis=(1, 2))[acc_order_msk]
        acc_order_price = self.agent_order_price[acc_order_msk]

        # calculate individual profit
        ind_rev_id_ub = self.individual_demand_pmf.shape[2]
        if not self.with_capacity_reservation:
            left_vac = self.capacity - agent_det_agg_consump
            left_vac[left_vac < 0] = 0
            reservation = np.minimum(
                np.repeat(
                    left_vac[:, :, np.newaxis],
                    ind_rev_id_ub,
                    axis=2
                ),
                np.resize(
                    np.arange(ind_rev_id_ub),
                    self.individual_demand_pmf.shape
                )
            )
        else:
            reservation = np.minimum(
                np.repeat(
                    capacity_reservation[:, :, np.newaxis],
                    ind_rev_id_ub,
                    axis=2
                ),
                np.resize(
                    np.arange(ind_rev_id_ub),
                    self.individual_demand_pmf.shape
                )
            )

        ind_profit = 0
        if self.with_capacity_reservation:
            compensation = 0
            if self.with_agent_cancel:
                # caculate agent profit here
                # FIXME for many accepted order, should make the agent event as
                # outer loop
                agent_profit_done = False

        # `reservation` is type x time x outcome
        for room_type, time_id, in product(
            range(self.num_type), range(self.time_span_len),
        ):
            for outcome_id in range(self.demand_ub[room_type] + 1):
                ind_arrival = reservation[room_type, time_id, outcome_id]
                ind_arrival_prob = self.individual_demand_pmf[
                    room_type, time_id, outcome_id
                ]
                if (ind_arrival_prob == 0):
                    continue  # invalid outcome or no need to count
                    # should not run to this block

                # individual profit would be zero without arrival
                # however compensation should count
                # if ind_arrival != 0:
                ind_cancel_probs = binom.pmf(
                    np.arange(ind_arrival + 1),
                    ind_arrival,
                    self.individual_cancel_rate[room_type]
                )
                ind_real = ind_arrival + np.dot(- np.arange(ind_arrival + 1),
                                                ind_cancel_probs)
                ind_profit += (
                    ind_arrival_prob *
                    ind_real *
                    self.individual_room_price[room_type]
                )

                # compensation
                if ((self.with_capacity_reservation) &
                    (not self.with_agent_cancel)):
                    compensation_amount = (
                        agent_det_agg_consump[room_type, time_id] +
                        ind_arrival -
                        np.arange(ind_arrival + 1) -
                        self.room_capacity[room_type]
                    )
                    compensation_amount[compensation_amount < 0] = 0
                    compensation += (
                        ind_arrival_prob *
                        self.compensation_price[room_type, time_id] *
                        np.dot(compensation_amount, ind_cancel_probs)
                    )

                elif ((self.with_capacity_reservation) & (self.with_agent_cancel)):
                    if not agent_profit_done:
                        agent_profit = 0
                    # only check uncertainty for accepted orders
                    for agent_cancel_outcome in product(
                        [0, 1], repeat=len(acc_order_upgrade_profit)
                    ):
                        agent_cancel_outcome = np.array(agent_cancel_outcome)
                        mask = np.hstack([agent_cancel_outcome,
                                        (1 - agent_cancel_outcome)])
                        event_prob_table = np.hstack([
                            acc_order_cancel_rate * agent_cancel_outcome,
                            (1 - acc_order_cancel_rate) * (1 - agent_cancel_outcome)
                        ])
                        agent_prob = np.prod(event_prob_table,
                                            where=mask.astype(bool))
                        compensation_amount = (
                            (acc_order_req[:, room_type, time_id] @
                             (1 - agent_cancel_outcome)) +
                            ind_arrival -
                            np.arange(ind_arrival + 1) -
                            self.room_capacity[room_type]
                        )
                        compensation_amount[compensation_amount < 0] = 0
                        compensation += (
                            ind_arrival_prob *
                            agent_prob *
                            self.compensation_price[room_type, time_id] *
                            np.dot(compensation_amount, ind_cancel_probs)
                        )
                        if not agent_profit_done:
                            agent_profit += (
                                ((acc_order_price + acc_order_upgrade_profit) @
                                (1 - agent_cancel_outcome)) * agent_prob
                            )

                    agent_profit_done = True

        # calculate agent profit
        if not self.with_agent_cancel:
            # agent profit
            agent_profit = acc_order_price.sum()
            agent_profit += acc_order_upgrade_profit.sum()

        elif not self.with_capacity_reservation:
            # agent order may cancel calculate compensation together
            agent_profit = 0
            for agent_cancel_outcome in product(
                [0, 1], repeat=len(acc_order_upgrade_profit)
            ):
                agent_cancel_outcome = np.array(agent_cancel_outcome)
                mask = np.hstack([agent_cancel_outcome,
                                  (1 - agent_cancel_outcome)])
                event_prob_table = np.hstack([
                    acc_order_cancel_rate * agent_cancel_outcome,
                    (1 - acc_order_cancel_rate) * (1 - agent_cancel_outcome)
                ])
                agent_prob = np.prod(event_prob_table,
                                     where=mask.astype(bool))
                agent_profit += agent_prob * (
                    np.dot(
                        (acc_order_upgrade_profit + acc_order_price),
                        (1 - agent_cancel_outcome)
                    )
                )

        obj_val = agent_profit + ind_profit
        if self.with_capacity_reservation:
            obj_val -= compensation
        return obj_val

    def get_df(self, order_acceptance, order_upgrade, capacity_reservation):
        acceptance_df = pd.DataFrame({'acceptance': order_acceptance})
        acceptance_df.index += 1
        index = pd.MultiIndex.from_product([
            np.arange(1, self.num_order + 1),
            np.arange(1, self.num_type + 1),
            np.arange(1, self.num_type + 1)
        ])
        upgrade_df = pd.DataFrame(order_upgrade.reshape(-1, 1), index=index)
        cap_rev_df = pd.DataFrame(capacity_reservation,
                                  columns=np.arange(
                                      1, self.time_span_len + 1
                                  ))
        cap_rev_df.index += 1
        return acceptance_df, upgrade_df, cap_rev_df

    def get_decision(self, mass_increment_depre_rate=0.8):
        start_time = time.perf_counter()
        # initialize decision variables
        order_acceptance = np.zeros(self.num_order)
        capacity_reservation = None

        # spare capacity for individual first
        order_level = self.get_order_level()

        # no any incentive to accept any inferior agent orders
        if (order_level < 0).all():
            upgrade_strategy = np.zeros(
                (self.num_order, self.num_type, self.num_type)
            )
            if self.with_capacity_reservation:
                capacity_reservation = (
                    self.capacity *
                    (1 / (1 - self.individual_cancel_rate)).reshape((-1, 1))
                )
            return order_acceptance, upgrade_strategy, capacity_reservation

        # if possible to extend requirement beyond capacity or
        # guarantee the requirement not to exceed capacity impact the decisions
        order_acceptance, order_upgrade, capacity_reservation = \
            self._get_decision(
                order_level,
                order_acceptance,
                depre_rate=mass_increment_depre_rate
            )
        self.calculation_time = time.perf_counter() - start_time

        return order_acceptance, order_upgrade, capacity_reservation

    # TODO may propose by rank and by grid and compare instead of a mix
    def _get_decision(self, order_level, order_acceptance, depre_rate):
        order_rank, value_ratio = self._get_order_rank()

        # calculate expected consumption to accumulate
        agent_req = np.zeros((self.num_type, self.time_span_len))
        step_size = int(depre_rate * self.num_order)
        num_acc = 0
        sorted_order_room = self.agent_order_room_quantity[order_rank, :]
        cal_order_defactor = 1 / (1 - self.cal_agent_cancel_rate)
        cal_order_stay = (self.agent_order_stay *
                          (1 - self.cal_agent_cancel_rate).reshape((-1, 1)))
        # already consider the cancellation rate
        sorted_cal_order_stay = cal_order_stay[order_rank, :]
        sorted_req = sorted_order_room[:, :, np.newaxis] * \
            sorted_cal_order_stay[: ,np.newaxis, :]
        sorted_price = self.agent_order_price[order_rank]

        # forward adding
        # accept some agent orders without upgrade within ideal amounts
        while True:
            if (agent_req <= order_level).all():  # all are under capacity
                agent_req += sorted_req[
                    num_acc: num_acc + step_size, :, :
                ].sum(axis=0)
                if ((agent_req > order_level).any() & (step_size == 1)):
                    break
                num_acc += step_size

            else:  # some exceed capacity
                if num_acc == 0:  # order_level are negative at the begining
                    break
                step_size = np.min([step_size, num_acc])
                agent_req -= sorted_req[
                    num_acc - step_size: num_acc, :, :
                ].sum(axis=0)
                num_acc -= step_size
                if ((agent_req <= order_level).all() & (step_size == 1)):
                    break

            step_size = int(depre_rate * step_size)
            step_size = step_size if step_size >= 1 else 1

        # num_good_order = (ratio > (1 - self.upgrade_fee_gap)).sum()
        # if num_good_order < num_acc:
        #     num_acc = num_good_order
        order_acceptance[order_rank[:num_acc]] = 1

        # add orders with upgrading
        acc_req = sorted_req[:num_acc, :, :].sum(axis=0)  # room x time
        # vaccancy left by individuals, we cannot upgrade negative grids
        # no any order causes negative grids
        left_vac = order_level - acc_req
        # left_vac[left_vac < 0] = 0  # underestimate shortage
        # however, since the compensation is linear, the underestimation may
        # not harm too much
        agent_profit = sorted_price[:num_acc].sum()
        order_upgrade = np.zeros(
            (self.num_order, self.num_type, self.num_type)
        )

        rest_sorted_req = sorted_req[num_acc:, :, :]
        rest_order_rank = order_rank[num_acc:]
        rest_sorted_price = sorted_price[num_acc:]
        acc_compensation = left_vac.copy()
        acc_compensation[acc_compensation > 0] = 0

        # extend the acceptance level by upgrade
        # and backward by reject first-phase orders
        # iterate by the extent of vaccancy
        # once the lowest vaccancy is getting bigger, we stop
        min_std_vac = np.std(left_vac)
        min_sum_vac = np.sum(left_vac)
        prioty = 0
        while True:

            # every time candidate was considered
            if prioty >= self.time_span_len:
                break

            # choose the time with most vaccancy to tune
            vac_rank = (- left_vac.sum(axis=0)).argsort()
            max_vac_time = vac_rank[prioty]

            # stopping criteria
            if (left_vac[:, max_vac_time] <= 0).all():
                # no need to try any worse time candidate
                break

            # possible to improve by adding orders
            candidates = np.where(
                rest_sorted_req[:, :, max_vac_time].sum(axis=1) > 0
            )[0]  # one dimension

            prev_fail = True  # if no candidate orders for target time
            for candidate_pos in candidates:

                target_order = rest_order_rank[candidate_pos]
                target_defactor = cal_order_defactor[target_order]
                valid_to_add = True
                # consider to add this order
                target_original_req = rest_sorted_req[candidate_pos]
                tmp_left_vac = left_vac - target_original_req
                tmp_acc_req = acc_req + target_original_req
                tmp_agent_profit = (agent_profit +
                                    rest_sorted_price[candidate_pos])
                target_order_upgrade = np.zeros((self.num_type, self.num_type,
                                                 self.time_span_len))

                # exceed_capacity = True
                # consider upgrading only extract those negativity caused by
                # this target order
                invalid_row, invalid_col = np.where((tmp_left_vac < 0) &
                                                    (target_original_req > 0))
                for room_pos, time_pos in zip(invalid_row, invalid_col):

                    grid_tune_done = False
                    if self.upgrade_rule in ['up', 'both']:

                        # consider to upgrade this order first
                        for to_type in range(room_pos + 1, self.num_type):
                            if tmp_left_vac[to_type, time_pos] < 0:
                                continue  # already negative
                            up_amount = np.min([
                                - tmp_left_vac[room_pos, time_pos],
                                tmp_left_vac[to_type, time_pos]
                            ])
                            tmp_acc_req[room_pos, time_pos] -= up_amount
                            tmp_acc_req[to_type, time_pos] += up_amount
                            tmp_left_vac[to_type, time_pos] -= up_amount
                            tmp_left_vac[room_pos, time_pos] += up_amount
                            target_order_upgrade[
                                room_pos, to_type, time_pos
                            ] += up_amount * target_defactor
                            tmp_agent_profit += (
                                self.upgrade_fee[room_pos, to_type] *
                                up_amount
                            )

                            if tmp_left_vac[room_pos, time_pos] >= 0:
                                grid_tune_done = True
                                break

                        # to need to consider upgrading other orders
                        # since the vacancies of up levels are squeezed out
                        # but maybe rejecting some orders may help

                        # if upgrade to upper levels cover all shortage
                        if grid_tune_done:
                            continue

                    if self.upgrade_rule in ['down', 'both']:

                        # consider to downgrade this order
                        for to_type in np.arange(room_pos)[::-1]:
                            if tmp_left_vac[to_type, time_pos] < 0:
                                continue  # already negative
                            down_amount = np.min([
                                - tmp_left_vac[room_pos, time_pos],
                                tmp_left_vac[to_type, time_pos]
                            ])
                            tmp_acc_req[room_pos, time_pos] -= down_amount
                            tmp_acc_req[to_type, time_pos] += down_amount
                            tmp_left_vac[to_type, time_pos] -= down_amount
                            tmp_left_vac[room_pos, time_pos] += down_amount
                            target_order_upgrade[
                                room_pos, to_type, time_pos
                            ] += down_amount * target_defactor
                            tmp_agent_profit += (
                                self.upgrade_fee[room_pos, to_type] *
                                down_amount
                            )

                            if tmp_left_vac[room_pos, time_pos] >= 0:
                                grid_tune_done = True
                                break

                        # no need to consider downgrading other orders

                    # ensure not exceeed capacity
                    if ((not grid_tune_done) &
                        (not self.with_capacity_reservation)):
                        if (tmp_acc_req[room_pos, time_pos] >
                            self.capacity[room_pos, time_pos]):
                            valid_to_add = False
                            break  # stop trying for next negative grid

                if not valid_to_add:
                    prev_fail = True
                    continue  # try next order

                # succeed to add this order
                # count in the compensation
                tmp_compensation = tmp_left_vac.copy()
                tmp_compensation[tmp_compensation > 0] = 0
                comp_room_amount = tmp_compensation - acc_compensation
                # `comp_room_amount` is negative
                tmp_agent_profit += (
                    comp_room_amount * self.compensation_price
                ).sum()

                if tmp_agent_profit < agent_profit:
                    prev_fail = True
                    continue  # consider another order

                prev_fail = False
                left_vac = tmp_left_vac
                acc_req = tmp_acc_req
                agent_profit = tmp_agent_profit
                acc_compensation = tmp_compensation
                # transfer flexible order upgrade into consistent along time
                order_upgrade[target_order] = (
                    target_order_upgrade.sum(axis=2) /
                    self.agent_order_stay[target_order].sum()
                )
                order_acceptance[target_order] = 1
                rest_order_rank = np.delete(rest_order_rank, candidate_pos,
                                            axis=0)
                rest_sorted_price = np.delete(rest_sorted_price, candidate_pos,
                                              axis=0)
                rest_sorted_req = np.delete(rest_sorted_req, candidate_pos,
                                            axis=0)
                prioty = 0
                # choose next time to tune again
                break

            if prev_fail:
                # fail to add any order at this time position
                prioty += 1
                # choose next target time to try again
                continue

            if (np.std(left_vac) < min_std_vac):
                min_std_vac = np.std(left_vac)
            if (np.sum(left_vac) < min_sum_vac):
                min_sum_vac = np.sum(left_vac)

            # FIXME may not be reached or unreasonable
            # lower std and lower sum of left vac is better
            # if ((np.std(left_vac) > min_std_vac) &
            #     (np.sum(left_vac) > min_sum_vac)):
            #     break

        # decided capacity reservation
        if self.with_capacity_reservation:
            ind_ub = get_ind_exp_req(self.scenario, self.time_span_len,
                                    self.with_ind_cancel)
            capacity_reservation = (
                (ind_ub + acc_compensation) *
                (1 / (1 - self.individual_cancel_rate)).reshape((-1, 1))
            )
            capacity_reservation[capacity_reservation < 0] = 0
        else:
            capacity_reservation = None
        return order_acceptance, order_upgrade, capacity_reservation

import configparser
scenarios = configparser.ConfigParser()
scenarios.read('scenarios.ini')
for scenario in scenarios.sections()[:1]:
    solver = Solver(scenarios[scenario], 3, 'up', 0, 0)
    order_acceptance, order_upgrade, capacity_reservation = \
        solver.get_decision()
     # convert to dataframe
    acceptance_df, upgrade_df, cap_rev_df = solver.get_df(
        order_acceptance,
        order_upgrade,
        capacity_reservation
    )
    obj_val = solver.get_obj(order_acceptance, order_upgrade,
                             capacity_reservation)