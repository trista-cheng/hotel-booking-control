import numpy as np
import xarray as xr
import pandas as pd

from scipy.stats import binom, bernoulli
from itertools import product

# For the room rate. Since there may be one order without any room requested,
# we generate times of batch size instances and filter those with at least one
# type required
OVERSAMPLE_RATIO = 5

def check_consistent_len(attr_list: list):
    is_valid = True
    length = len(attr_list[0])
    for attr in attr_list[1:]:
        if len(attr) != length:
            is_valid = False
            break
    return is_valid

class DataGenerator:
    def __init__(self, time_span_len: int, num_room_type: int,
                 capacity: np.array, individual_price: np.array,
                 upgrade_fee_gap_multiplier: float,
                 compensation_price: np.array) -> None:
        """
        Lengths of capacity, individual_price, individual_success_rate,
        individual_pop_size should be the same.
        They imply the number of room type.

        Caution
        -------
        Values in all index sets starts from 0 and are integer.

        Parameters
        ----------
        time_span_len(int): The total time length.
        capacity(1D array): The capacity of each room type.
        inidividual_price(1D array): The price for each room type.
        upgrade_fee_gap_multiplier: include upgrade and downgrade.
        """
        ## validate
        # check for room type
        is_valid = check_consistent_len(
            [capacity, individual_price]
        )
        if not is_valid:
            raise Exception("Length is not consistant along room type.")

        self.time_span_len = int(time_span_len)
        self.time_span = np.arange(time_span_len)
        self.num_room_type = num_room_type
        self.room_type_set = np.arange(num_room_type)
        self.capacity = np.array(capacity)
        self.individual_price = np.array(individual_price)
        self.upgrade_fee_gap_multiplier = upgrade_fee_gap_multiplier
        self.compensation_price = compensation_price
        # TODO upgrade fee gap here is somehow nonsense, hotel info may better

    def generate_hotel_info(self):
        """
        Returns
        ---------
        capacity, individual room price, upgrade fee
        """
        # 2D array
        upgrade_diff = -(self.individual_price.reshape((-1, 1)) -
                         self.individual_price)
        upgrade_fee = (
            np.triu(upgrade_diff) * (1 - self.upgrade_fee_gap_multiplier) +
            np.tril(upgrade_diff) * (1 + self.upgrade_fee_gap_multiplier)
        )
        compensation_price = np.repeat(
            self.compensation_price.reshape((-1, 1)),
            self.time_span_len,
            axis=1
        )
        return (self.capacity, self.individual_price, upgrade_fee,
                compensation_price)

    def generate_agent_order(self, room_request_ratio_threshold: float,
                             avg_stay_duration: int, avg_num_room: np.array,
                             padding_rate: float, room_rate: np.array,
                             price_multiplier: float, batch_size: int,
                             avg_cancel_rate: float, **kwargs):
        # TODO price and upgrade fee and padding are less flexible and variety,
        # static to multiplier.

        """
        Parameter
        ---------
        room_request_ratio_threshold(float): The request quantity must
            exceed capacity multiplied by this number.
        avg_stay_duration(int): Average stay duration(regardless of room type).
            num_order(int): Number of order.
        avg_num_room(1D array): average number of rooms for each type
            in one order.
        padding_rate([0, 1]): Proportion to expand from average value for
            uniform distribution simultation.
        room_rate(1D): Average probability to book given type in an order.
        price_multiplier([0, 1]): The price for an agent order is calculated by
            the individual price and multiplier
        batch_size: determine the number orders generated iteratively.
        avg_cancel_rate: uniform distribution with avg and padding.

        Returns
        --------
        agent_order_price
        agent_order_room_quantity
        agent_order_stay
        agent_cancel_dict: dictionary of PMF containing outcome of each order
            and the corresponding prob
        agent_cancel_prob: np.array of probability for each outcome
        agent_cancel_outcome: np.array(outcome x order) of possible outcomes
        """

        agent_order_stay_pool = np.empty((0, self.time_span_len))
        agent_order_room_quantity_pool = np.empty((0, self.num_room_type))
        agent_order_price_pool = np.empty((0))
        while True:
            ## stay duration
            # start time
            start_period = np.random.choice(self.time_span, batch_size,
                                            replace=True)
            # duration length
            stay_ub = int(np.ceil((1 + padding_rate) * avg_stay_duration))
            stay_lb = int(np.floor((1 - padding_rate) * avg_stay_duration))
            stay_lb = 1 if stay_lb < 1 else stay_lb
            rng = np.random.default_rng()
            stay_len = rng.integers(low=stay_lb, high=stay_ub, endpoint=True,
                                    size=batch_size)
            agent_order_stay = np.zeros(
                (batch_size, self.time_span_len + stay_ub),
                dtype=np.int8
            )
            for order_id in range(batch_size):
                agent_order_stay[
                    order_id,
                    start_period[order_id]:
                    start_period[order_id] + stay_len[order_id]
                ] = 1
            agent_order_stay = agent_order_stay[:,:self.time_span_len]
            stay_len = agent_order_stay.sum(axis=1)
            if (stay_len <= 0).any():
                print("Nonsense order appear!")

            ## number of rooms in orders
            # TODO 0 for all type in one order
            order_room_bin = bernoulli.rvs(
                p=room_rate,
                size=(batch_size * OVERSAMPLE_RATIO, self.num_room_type)
            )
            order_room_bin = order_room_bin[order_room_bin.sum(axis=1) > 0]
            order_room_bin = order_room_bin[np.random.choice(
                range(order_room_bin.shape[0]), size=batch_size, replace=False
            )]

            num_room_ub = np.ceil((1 + padding_rate) * avg_num_room)
            num_room_lb = np.floor((1 - padding_rate) * avg_num_room)
            num_room_ub = np.array([int(ub) for ub in num_room_ub])
            num_room_lb = np.array([int(lb) for lb in num_room_lb])
            num_room_lb[num_room_lb < 1] = 1
            rng = np.random.default_rng()
            quantity = rng.integers(low=num_room_lb, high=num_room_ub,
                                    endpoint=True,
                                    size=(batch_size, self.num_room_type))
            agent_order_room_quantity = quantity * order_room_bin
            if (agent_order_room_quantity.sum(axis=1) <= 0).any():
                print("Nonsense order appear!")

            # 1D array
            price_mul_ub = (1 + padding_rate) * price_multiplier
            price_mul_lb = (1 - padding_rate) * price_multiplier
            rng = np.random.default_rng()
            price_mul = rng.uniform(low=price_mul_lb, high=price_mul_ub,
                                   size=batch_size)
            # endpoint is exclusive
            agent_order_price = np.dot(
                stay_len.reshape(-1, 1) * agent_order_room_quantity,
                self.individual_price
            ) * price_mul
            # FIXME NAME BAD. price_mul mean_price_mul

            agent_order_stay_pool = np.concatenate(
                [agent_order_stay_pool, agent_order_stay],
                axis=0
            )
            agent_order_room_quantity_pool = np.concatenate(
                [agent_order_room_quantity_pool, agent_order_room_quantity],
                axis=0
            )
            agent_order_price_pool = np.concatenate(
                [agent_order_price_pool, agent_order_price],
                axis=0
            )
            room_request = np.dot(
                agent_order_room_quantity_pool.T, agent_order_stay_pool
            )  # .mean(axis=1)
            # room_request: room x time
            room_request_ratio = (room_request.sum() /
                                  (self.capacity.sum() * self.time_span_len))
            if room_request_ratio > room_request_ratio_threshold:
                break
        cancel_rate_ub = avg_cancel_rate * (1 + padding_rate)
        cancel_rate_lb = avg_cancel_rate * (1 - padding_rate)
        rng = np.random.default_rng()
        num_order = len(agent_order_price_pool)
        agent_cancel_rate = rng.uniform(low=cancel_rate_lb, high=cancel_rate_ub,
                                        size=num_order)
        return (agent_order_price_pool, agent_order_room_quantity_pool,
                agent_order_stay_pool, agent_cancel_rate)

    def generate_individual(self, individual_success_rate: np.array,
                            individual_pop_size: np.array,
                            cancel_rate: np.array, **kwargs):
        """
        Store prob data for numpy compatible use. NO quantity data.

        Parameters
        ----------
        individual_success_rate(2D array): The success rate of each room type in
        each time period. The dimension is #room type * time_span
        individual_pop_size(1D array): The population size of each room type.
        cancel_rate(1D array)

        Return
        --------
        pmf_dict_tuple_key(dict):. Keys are room type id, time period id,
        `"quantity"` & `"prob"`
        demand_ub(1D array): It is same as `individual_pop_size`.
        cancel_rate: the cancel rate of each room type
        """
        # TODO generate useless prob
        # For each room type in each period, calculate same range of possible
        # outcome. Some impossible outcomes are still recorded with 0 prob.
        pmf = np.array([
            binom.pmf(
                i,
                individual_pop_size.reshape((-1, 1)),
                individual_success_rate
            )
            for i in range(int(individual_pop_size.max()) + 1)
        ]).swapaxes(0, 1).swapaxes(1, 2)
        # Keep low coupling. For possible realization value for demand quantity,
        # the possible maximum value is the population value. Do not need to
        # consider possible maximum effective value, which means we should not
        # take the capacity into account in PMF calculation.

        # FIXME: NO numpy work here
        # decide to store as pandas or xarray
        # xr.DataArray(pmf).to_netcdf(f"{file_name}.nc")
        data = xr.DataArray(
            pmf,
            dims=("room", "time", "outcome"),
            coords={
                "room": np.arange(self.num_room_type) + 1,
                "time": np.arange(self.time_span_len) + 1,
                "outcome": np.arange(pmf.shape[2]) + 1,
                "quantity": ("outcome", np.arange(pmf.shape[2]))
            }
        )
        pmf_df = data.to_dataframe(name="prob")
        # df_index = pd.MultiIndex.from_product([
        #     (np.arange(self.num_room_type) + 1).astype(str),
        #     (np.arange(self.time_span_len) + 1).astype(str),
        #     (np.arange(pmf.shape[2]) + 1).astype(str)
        # ])
        # pmf_df = pd.DataFrame(
        #     {
        #         'quantity':
        #         np.resize(
        #             np.arange(pmf.shape[2]),
        #             self.num_room_type * self.time_span_len * pmf.shape[2],
        #         ),
        #         'prob': pmf
        #     },
        #     index=df_index
        # )
        pmf_dict_tuple_key = pmf_df.to_dict(orient='index')

        return pmf_dict_tuple_key, individual_pop_size, cancel_rate
