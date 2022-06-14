"""
NOT maintaining anymore.
Be carful to use it.
"""

import numpy as np

from data_reader import CSVDataReader

class OrderManager:

    def __init__(self, scenario, instance_id, upgrade_rule):
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
        self.upgrade_rule = upgrade_rule
        self.scenario = scenario

    def get_order_acceptance(self, with_capacity_reservation, with_agent_cancel,
                             depre_rate=0.8):
        cost = np.dot(
            self.agent_order_room_quantity,
            self.individual_room_price.reshape((-1, 1))
        ).flatten() * self.agent_order_stay.sum(axis=1)
        ratio = (self.agent_order_price / cost)
        if with_agent_cancel:
            ratio *= (1 - self.agent_cancel_rate)
        capacity = (self.room_capacity.reshape((-1, 1)) *
                    np.ones(self.time_span_len))
        rank = (- ratio).argsort()
        consump = np.zeros((self.num_type, self.time_span_len))
        step_size = int(depre_rate * self.num_order)
        num_acc = 0
        sorted_order_room = self.agent_order_room_quantity[rank, :]
        sorted_room_order = sorted_order_room.T
        sorted_order_time_stay = self.agent_order_stay[rank, :]
        while True:
            if (consump <= capacity).all():
                consump += np.dot(
                    sorted_room_order[:, num_acc: num_acc+step_size],
                    sorted_order_time_stay[num_acc: num_acc+step_size, :]
                )
                if ((consump > capacity).any() & (step_size == 1)):
                    break
                num_acc += step_size

            else:
                step_size = np.min([step_size, num_acc])
                consump -= np.dot(
                    sorted_room_order[:, num_acc-step_size: num_acc],
                    sorted_order_time_stay[num_acc-step_size: num_acc, :]
                )
                num_acc -= step_size
                if ((consump <= capacity).all() & (step_size == 1)):
                    break

            step_size = int(depre_rate * step_size)
            step_size = step_size if step_size >= 1 else 1

        acc_pos_set = rank[:num_acc]
        order_acceptance = {}
        order_set = []
        for order_pos in acc_pos_set:
            order_acceptance[str(order_pos + 1)] = 1
            order_set.append(str(order_pos + 1))

        rest_order_pos = rank[num_acc:]
        rest_req = sorted_order_room[num_acc:, :, np.newaxis] * \
            sorted_order_time_stay[num_acc: ,np.newaxis, :]
        left_vac = capacity - np.dot(
            sorted_room_order[:, :num_acc],
            sorted_order_time_stay[:num_acc, :]
        )

        # extend by non-upgraded orders
        for iter_id in range(len(rest_order_pos)):
            valid_msk = (rest_req <= left_vac).all(axis=(1, 2))
            # TODO may set an UB exceeding capacity for probability of cancel.
            # Or consider upgrades.
            if valid_msk.sum() == 0:
                break
            sorted_acc_pos = valid_msk.argmax()
            left_vac -= rest_req[sorted_acc_pos, :, :]
            if (left_vac < 0).any():
                raise Exception("Nonsense!")
            acc_order_pos = rest_order_pos[sorted_acc_pos]
            order_acceptance[str(acc_order_pos + 1)] = 1
            order_set.append(str(acc_order_pos + 1))
            rest_order_pos = np.delete(rest_order_pos, sorted_acc_pos)
            rest_req = np.delete(rest_req, sorted_acc_pos, axis=0)

        # extend the acceptance level by upgrade
        for iter_id in range(len(rest_order_pos)):
            # TODO until every order is considered NOT consumption reach some
            # threshold
            tmp_left_vac = left_vac - rest_req[iter_id, :, :]
            if (tmp_left_vac.sum(axis=0) < 0).any():
                continue
            # must valid for 'both' upgrades rule or with cpaacity reservation
            # may invalid for other cases
            valid_to_add = True
            invalid_row, invalid_col = np.where(tmp_left_vac < 0)
            for room_pos, time_pos in zip(invalid_row, invalid_col):

                if ((not with_capacity_reservation) &
                    (self.upgrade_rule != 'both')):
                    if self.upgrade_rule == "up":
                        if tmp_left_vac[room_pos:, time_pos].sum() < 0:
                            valid_to_add = False
                            break

                        from_type = room_pos
                        for to_type in range(room_pos + 1, self.num_type):
                            tmp_left_vac[to_type, time_pos] += \
                                tmp_left_vac[from_type, time_pos]
                            tmp_left_vac[from_type, time_pos] = 0
                            if tmp_left_vac[to_type, time_pos] >= 0:
                                break
                            from_type = to_type
                    else:  # 'down'
                        if tmp_left_vac[:(room_pos + 1), time_pos].sum() < 0:
                            valid_to_add = False
                            break

                        to_type = room_pos
                        for from_type in np.arange(room_pos)[::-1]:
                            tmp_left_vac[from_type, time_pos] += \
                                tmp_left_vac[to_type, time_pos]
                            tmp_left_vac[to_type, time_pos] = 0
                            if tmp_left_vac[from_type, time_pos] >= 0:
                                break
                            to_type = from_type

                else:
                    if tmp_left_vac[:(room_pos + 1), time_pos].sum() >= 0:
                        to_type = room_pos
                        for from_type in np.arange(room_pos)[::-1]:
                            tmp_left_vac[from_type, time_pos] += \
                                tmp_left_vac[to_type, time_pos]
                            tmp_left_vac[to_type, time_pos] = 0
                            if tmp_left_vac[from_type, time_pos] >= 0:
                                break
                            to_type = from_type
                    else:
                        partial_up = tmp_left_vac[:room_pos, time_pos].sum()
                        tmp_left_vac[:room_pos, time_pos] = 0
                        tmp_left_vac[room_pos, time_pos] += partial_up

                        from_type = room_pos
                        for to_type in range(room_pos + 1, self.num_type):
                            tmp_left_vac[to_type, time_pos] += \
                                tmp_left_vac[from_type, time_pos]
                            tmp_left_vac[from_type, time_pos] = 0
                            if tmp_left_vac[to_type, time_pos] >= 0:
                                break
                            from_type = to_type

            if valid_to_add:
                left_vac = tmp_left_vac
                order_set.append(str(rest_order_pos[iter_id] + 1))
                order_acceptance[str(rest_order_pos[iter_id] + 1)] = 1

        # a.ravel()[:,newaxis]*b.ravel()[newaxis,:]
        # [np.outer(x[i],y[i]) for i in range(x.shape[0])]

        return order_set, order_acceptance
