from asyncore import read
import numpy as np

from data_reader import CSVDataReader


class ConstraintViolation(Exception):
    def __init__(self, scenario, instance_id, hint=None, *args: object) -> None:
        self.scenario = scenario
        self.instance_id = instance_id
        self.hint = hint
        super().__init__(*args)
    
    def __str__(self):
        return f"{self.hint}| Scenario: {self.scenario}| \
                 Instance: {self.instance_id}"

class Validator:
    def __init__(self, scenario, instance_id: int, acceptance: np.array, 
                 upgrade: np.array):
        """
        Methods
        --------
        `validate_shape`
        `validate_capacity`
        `validate_obj`
        """
        self.scenario = scenario
        self.instance_id = instance_id
        self.acceptance = acceptance
        self.upgrade = upgrade
        reader = CSVDataReader(scenario)
        self.order_price, self.order_room, self.order_stay = \
            reader.collect_agent_info(instance_id)
        self.capacity, self.upgrade_fee = reader.collect_hotel_info(instance_id)
        self.ind_pmf, self.ind_room_price = reader.collect_individual_info()
        self.error_base = ConstraintViolation(self.scenario, self.instance_id)
        self.num_room = self.capacity.shape[0]
        self.num_order = self.order_price.shape[0]
        self.num_period = self.order_stay.shape[1]
        
    def validate_shape(self):
        if not (self.num_order == self.acceptance.shape[0]):
            self.error_base.hint = "Acceptance length not order length"
            raise self.error_base
        if not (
            (self.upgrade.shape[1] == self.upgrade.shape[2]) &
            (self.upgrade.shape[1] == self.num_room) &
            (self.upgrade.shape[0] == self.num_order)
        ):
            self.error_base.hint = "Upgrade shape error"
            raise self.error_base
        if (self.upgrade.sum(axis=0).diagonal() != 0).any():
            self.error_base.hint = "Nonzero in diagonal of upgrade"
            raise self.error_base

    def _validate_capacity(self):
        
        upgrade_diff = (
            np.dot(self.upgrade, -np.ones((self.num_room, 1))).reshape(
                (self.num_order, self.num_room)
            ) +
            np.dot(np.ones(self.num_room), self.upgrade).reshape(
                (self.num_order, self.num_room)
            )
        )
        final_demand = np.dot(
            (self.order_room + upgrade_diff).T, 
            self.acceptance.reshape((self.num_order, 1)) * self.order_stay
        )
        if (final_demand > 
            np.repeat(
                np.reshape(self.capacity, (self.num_room, 1)), 
                self.num_period,
                axis=1
            )).any():
            self.error_base.hint = "Demand exceeds capacity"
            raise self.error_base
        self.vacancy = (self.capacity.reshape((self.num_room, 1)) * 
                        np.ones(self.num_period))
        self.vacancy = self.vacancy - final_demand

    def validate_capacity_obj(self, obj):
        self._validate_capacity()
        
        agent_rev = np.dot(
            self.acceptance, 
            self.order_price.reshape((self.num_order, 1))
        )
        upgrade_rev = (
            self.upgrade * 
            self.upgrade_fee * 
            self.order_stay.sum(axis=1).reshape(self.num_order, 1, 1)
        )
        agent_rev = agent_rev + upgrade_rev.sum()

        vacancy = self.vacancy.reshape((self.num_room, self.num_period, 1))
        vacancy = vacancy.repeat(self.ind_pmf.shape[2], 2).reshape(
            (self.num_room, self.num_period, self.ind_pmf.shape[2], 1)
        )
        ind_quantity = np.resize(np.arange(self.ind_pmf.shape[2]),
            (self.num_room, self.num_period, self.ind_pmf.shape[2], 1)
        )
        sale = np.amin(np.concatenate([vacancy, ind_quantity], axis=3), axis=3)
        ind_rev = sale * self.ind_pmf
        total_exp_rev = agent_rev + ind_rev.sum()

        if not np.isclose(total_exp_rev, obj):
            self.error_base.hint = "Obj error"
            raise self.error_base 
        



        

