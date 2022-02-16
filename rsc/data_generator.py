import numpy as np
import xarray as xr

from scipy.stats import binom, bernoulli


def check_consistant_len(attr_list: list):
    is_valid = True
    length = len(attr_list[0])
    for attr in attr_list[1:]:
        if len(attr) != length:
            is_valid = False
            break
    return is_valid

class DataGenerator:
    def __init__(self, time_span_len: int, capacity: np.array,
                 individual_price: np.array,) -> None:
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
        """
        ## validate
        # check for room type
        is_valid = check_consistant_len([capacity, individual_price, ])
        if not is_valid:
            raise Exception("Length is not consistant along room type.")

        self.time_span_len = int(time_span_len)
        self.time_span = np.arange(time_span_len)
        self.num_room_type = len(capacity)
        self.room_type_set = np.arange(0, len(capacity))
        self.capacity = np.array(capacity)
        self.individual_price = np.array(individual_price)


    def generate_agent_order(self, avg_stay_duration: int, num_order: int,
                             avg_num_room: np.array, padding_rate: float,
                             capacity_multiplier: np.array,
                             price_mutiplier: float,
                             upgrade_fee_multiplier: float):
        # TODO price and upgrade fee and padding are less flexible

        """
        Parameter
        ---------
        avg_stay_duration(int): Average stay duration(regardless of room type).
        num_order(int): Number of order.
        avg_num_room(1D): Average quantity in orders with request to given
        room type.
        padding_rate([0, 1]): Proportion to expand from average value for
        uniform distribution simultation.
        room_rate(1D): Average probability to book given type in an order.
        capacity_multiplier(1D): The product of stay duration, #order,
        probability of request for given room, and #room equals this given
        constant to mutiplied by capacity and time length.
        price_mutiplier([0, 1]): The price for an agent order is calculated by
        the individual price and multiplier
        upgrade_fee_mutiplier([0, 1]): Same as price_multiplier.
        """

        # 1D array
        room_rate = (
            (capacity_multiplier * self.capacity * self.time_span_len) /
            (num_order * avg_stay_duration * avg_num_room)
        )
        if (room_rate > 1).any():
            room_rate = room_rate / room_rate.sum()
        # FIXME 2 is another magic number
        room_rate *= 2
        room_rate[room_rate > 1] = 1

        ## stay duration
        # start time
        start_period = np.random.choice(self.time_span, num_order, replace=True)
        # duration length
        stay_ub = int(np.ceil((1 + padding_rate) * avg_stay_duration))
        stay_lb = int(np.floor((1 - padding_rate) * avg_stay_duration))
        stay_lb = 0 if stay_lb < 0 else stay_lb
        rng = np.random.default_rng()
        stay_len = rng.integers(low=stay_lb, high=stay_ub, endpoint=True,
                            size=num_order)
        agent_order_stay = np.zeros((num_order, self.time_span_len + stay_ub),
                                    dtype=np.int8)
        for order_id in range(num_order):
            agent_order_stay[
                order_id,
                start_period[order_id]:
                start_period[order_id] + stay_len[order_id]
            ] = 1
        agent_order_stay = agent_order_stay[:,:self.time_span_len]
        stay_len = agent_order_stay.sum(axis=1)

        ## number of rooms in orders
        # FIXME 1000 magic number
        order_room_bin = bernoulli.rvs(p=room_rate, size=(1000, self.num_room_type))
        # order_room_bin = order_room_bin[order_room_bin.sum(axis=1) > 0]
        # while order_room_bin.shape[0] < num_order:
        #     tmp = bernoulli.rvs(p=room_rate, size=(100, self.num_room_type))
        #     order_room_bin = np.concatenate(
        #         [order_room_bin, tmp[tmp.sum(axis=1) > 0]]
        #     )
        # order_room_bin = order_room_bin[
        #     np.random.choice(np.arange(len(order_room_bin)),
        #     size=num_order, replace=False)
        # ]

        num_room_ub = np.ceil((1 + padding_rate) * avg_num_room)
        num_room_lb = np.floor((1 - padding_rate) * avg_num_room)
        num_room_ub = np.array([int(ub) for ub in num_room_ub])
        num_room_lb = np.array([int(lb) for lb in num_room_lb])
        num_room_lb[num_room_lb < 0] = 0
        rng = np.random.default_rng()
        # FIXME 1000 magic number
        quantity = rng.integers(low=num_room_lb, high=num_room_ub,
                                endpoint=True,
                                size=(1000, self.num_room_type))
        agent_order_room_quantity = quantity * order_room_bin
        agent_order_room_quantity = agent_order_room_quantity[
            agent_order_room_quantity.sum(axis=1) > 0
        ]
        agent_order_room_quantity = agent_order_room_quantity[
            np.random.choice(np.arange(len(agent_order_room_quantity)),
            size=num_order, replace=False)
        ]

        # 1D array
        agent_order_price = np.dot(stay_len.reshape(-1, 1) * agent_order_room_quantity,
                                   self.individual_price) * price_mutiplier
        # 2D array
        upgrade_fee = (self.individual_price.reshape(-1, 1) -
                       self.individual_price) * upgrade_fee_multiplier

        return (agent_order_price, agent_order_room_quantity, agent_order_stay,
                upgrade_fee)

    def generate_individual(self, individual_success_rate: np.array,
                            individual_pop_size: np.array, file_name: str):
        """
        Store prob data for numpy compatible use. NO quantity data.

        Parameters
        ----------
        individual_success_rate(2D array): The success rate of each room type in
        each time period. The dimension is #room type * time_span
        individual_pop_size(1D array): The population size of each room type.

        Return
        --------
        pmf_dict(dict):. Keys are room type id, time period id,
        `"quantity"` & `"prob"`
        """
        # FIXME generate useless prob
        # For each room type in each period, calculate same range of possible
        # outcome. Some impossible outcomes are still recorded with 0 prob.
        pmf = np.array([
            binom.pmf(
                i,
                individual_pop_size.reshape((-1, 1)),
                individual_success_rate
            )
            for i in range(int(
                # np.min(self.capacity.max(), individual_pop_size.max()) + 1
                # FIXME 100 is magic number
                100
            ))
        ]).swapaxes(0, 1).swapaxes(1, 2)

        xr.DataArray(pmf).to_netcdf(f"{file_name}.nc")
        # xr.open_dataset("saved_on_disk.nc").to_numpy()
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
        pmf_dict = data.to_dataframe(name="prob").T.to_dict()
        return pmf_dict
