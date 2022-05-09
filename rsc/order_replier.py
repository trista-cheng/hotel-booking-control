from matplotlib.pyplot import step
import numpy as np

class OrderManager:

    # FIXME use data reader as argument
    def __init__(self, time_span, room_type_set, room_capacity, 
                 agent_order_price, agent_order_room_quantity, agent_order_stay, 
                 agent_cancel_rate, individual_room_price):
        self.time_span_len = len(time_span)
        self.num_order = len(agent_order_price)
        self.num_type = len(room_type_set)
        self.room_capacity = np.array(list(room_capacity.values()))
        self.agent_order_price = np.array(list(agent_order_price.values()))
        agent_order_room_quantity_np = np.zeros(
            (self.num_order, self.num_type)
        )
        for order_pos in range(self.num_order):
            for type_pos in range(self.num_type):
                agent_order_room_quantity_np[order_pos, type_pos] =\
                    agent_order_room_quantity[str(order_pos+1)][str(type_pos+1)]
        self.agent_order_room_quantity = agent_order_room_quantity_np
        agent_order_stay_np =  np.zeros((self.num_order, self.time_span_len))
        for order_pos in range(self.num_order):
            for t_pos in range(self.time_span_len):
                agent_order_stay_np[order_pos, t_pos] =\
                    agent_order_stay[str(order_pos+1)][str(t_pos+1)]
        self.agent_order_stay = agent_order_stay_np
        self.agent_cancel_rate = np.array(list(agent_cancel_rate.values()))
        self.individual_room_price = np.array(list(individual_room_price.values()))

    def get_order_acceptance(self, depre_rate=0.8):
        cost = np.dot(
            self.agent_order_room_quantity, 
            self.individual_room_price.reshape((-1, 1))
        ).flatten() * self.agent_order_stay.sum(axis=1)
        ratio = (self.agent_order_price / cost) * (1 - self.agent_cancel_rate)
        capacity = self.room_capacity.reshape((-1, 1)) * np.ones(self.time_span_len)
        rank = (- ratio).argsort()
        consump = np.zeros((self.num_type, self.time_span_len))
        step_size = int(depre_rate * self.num_order)
        num_acc = 0
        sorted_room_order = self.agent_order_room_quantity[rank, :].T
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
        for order_pos in range(self.num_order):
            if order_pos in acc_pos_set:
                order_acceptance[str(order_pos + 1)] = 1
                order_set.append(str(order_pos + 1))

        # for order_pos in range(self.num_order):
        #     if order_pos in acc_pos_set:
        #         order_acceptance[str(order_pos + 1)] = 1
        #     else:
        #         order_acceptance[str(order_pos + 1)] = 0
        
        return order_set, order_acceptance

        

