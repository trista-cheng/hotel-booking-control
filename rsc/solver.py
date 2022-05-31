import numpy as np

from data_reader import CSVDataReader
from tools import get_ind_exp_req

BASE_PARAM = 'settings/base.npy'

class Solver:

    def __init__(self, scenario: dict, instance_id: int, upgrade_rule: str,
                 with_capacity_reservation: bool, with_agent_cancel: bool,
                 with_ind_cancel=True):
        """_summary_

        Args:
            scenario (dict): _description_
            instance_id (int): _description_
            upgrade_rule (str): available strings are `up`, `down`, and `both`
            with_capacity_reservation (bool): _description_
            with_agent_cancel (bool): _description_
            with_ind_cancel (bool, optional): _description_. Defaults to True.
        """
        reader = CSVDataReader(scenario=scenario)
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


    @property
    def get_calculation_time(self):
        return self.calculation_time

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

    def get_obj(self):
        return

    def validate(self, order_acceptance, upgrade_strategy):
        return

    def get_decision(self, mass_increment_depre_rate=0.8):
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
        order_acceptance, upgrade_strategy = self._get_decision(
            order_level,
            order_acceptance,
            depre_rate=mass_increment_depre_rate
        )
        return order_acceptance, upgrade_strategy

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
            if prioty > self.time_span_len:
                break

            # choose the time with most vaccancy to tune
            vac_rank = (- left_vac.sum(axis=0)).argsort()
            max_vac_time = vac_rank[prioty]

            # stopping criteria
            if (left_vac[:, max_vac_time] <= 0).all():
                prioty += 1  # try next time candidate
                continue

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
                target_order_upgrade = np.zeros((self.num_type, self.num_type))

                # exceed_capacity = True
                # consider upgrading only extract those negativity caused by
                # this target order
                invalid_row, invalid_col = np.where((tmp_left_vac < 0) &
                                                    (target_original_req > 0))
                for room_pos, time_pos in zip(invalid_row, invalid_col):

                    if self.upgrade_rule in ['up', 'both']:

                        # consider to upgrade this order first
                        shortage = - tmp_left_vac[room_pos, time_pos]
                        up_total_ub = tmp_left_vac[room_pos+1:, time_pos].sum()
                        if shortage > up_total_ub:  # need more upgrades

                            if ((self.upgrade_rule == 'up') &
                                (not self.with_capacity_reservation) &
                                (tmp_acc_req > self.capacity).any()):
                                valid_to_add = False
                                break

                            # modify the target position
                            tmp_acc_req[room_pos, time_pos] -= up_total_ub
                            tmp_left_vac[room_pos, time_pos] += up_total_ub

                            # modify the up level info
                            up_amounts = tmp_left_vac[room_pos + 1: , time_pos]
                            target_order_upgrade[
                                room_pos, room_pos + 1:
                            ] += up_amounts * target_defactor
                            tmp_acc_req[room_pos + 1: , time_pos] += up_amounts
                            tmp_left_vac[room_pos + 1: , time_pos] = 0
                            tmp_agent_profit += np.dot(
                                self.upgrade_fee[room_pos][room_pos + 1:],
                                up_amounts
                            )

                        else:  # finish upgrades for this grid
                            from_type = room_pos
                            for to_type in range(room_pos + 1, self.num_type):

                                up_amount = - tmp_left_vac[from_type, time_pos]
                                tmp_acc_req[from_type, time_pos] -= up_amount
                                tmp_acc_req[to_type, time_pos] += up_amount
                                tmp_left_vac[to_type, time_pos] -= up_amount
                                tmp_left_vac[from_type, time_pos] += up_amount
                                target_order_upgrade[
                                    room_pos, to_type
                                ] += up_amount * target_defactor
                                tmp_agent_profit += (
                                    self.upgrade_fee[room_pos,to_type] *
                                    up_amount
                                )

                                if tmp_left_vac[to_type, time_pos] >= 0:
                                    continue
                                from_type = to_type

                            continue  # next invalid grid

                        # to need to consider upgrading other orders
                        # since the vacancies of up levels are squeezed out
                        # but maybe rejecting some orders may help

                    elif self.upgrade_rule in ['down', 'both']:

                        # consider to downgrade this order
                        shortage = - tmp_left_vac[room_pos, time_pos]
                        down_total_ub = tmp_left_vac[:room_pos, time_pos].sum()
                        to_type = room_pos
                        if shortage > down_total_ub:

                            if (not self.with_capacity_reservation):
                                valid_to_add = False
                                break

                            tmp_acc_req[room_pos, time_pos] -= down_total_ub
                            tmp_left_vac[room_pos, time_pos] += down_total_ub

                            down_amounts = tmp_left_vac[:room_pos, time_pos]
                            tmp_acc_req[:room_pos, time_pos] += down_amounts
                            tmp_left_vac[:room_pos, time_pos] = 0
                            target_order_upgrade[
                                room_pos, :room_pos
                            ] += down_amounts * target_defactor
                            tmp_agent_profit += np.dot(
                                self.upgrade_fee[room_pos, :room_pos],
                                down_amounts
                            )
                        else:
                            from_type = room_pos
                            for to_type in np.arange(room_pos)[::-1]:
                                down_amount = - tmp_left_vac[from_type, time_pos]
                                tmp_acc_req[from_type, time_pos] -= down_amount
                                tmp_acc_req[to_type, time_pos] += down_amount
                                tmp_left_vac[from_type, time_pos] += down_amount
                                tmp_left_vac[to_type, time_pos] -= down_amount
                                tmp_agent_profit += (
                                    self.upgrade_fee[room_pos, to_type] *
                                    down_amount
                                )
                                target_order_upgrade[
                                    room_pos, to_type
                                ] += down_amount * target_defactor
                                if tmp_left_vac[to_type, time_pos] >= 0:
                                    continue
                                from_type = to_type
                        # no need to consider downgrading other orders

                if not valid_to_add:
                    prev_fail = True
                    continue

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
                order_upgrade[target_order] = target_order_upgrade
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


        return order_acceptance, order_upgrade

import configparser
scenarios = configparser.ConfigParser()
scenarios.read('scenarios.ini')
for scenario in scenarios.sections():
    solver = Solver(scenarios[scenario], 0, 'up', True, True)
    acceptance, upgrade, capacity_reservation = solver.get_decision()
