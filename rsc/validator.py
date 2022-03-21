import numpy as np
import pandas as pd

from data_reader import CSVDataReader
from data_manager import OUTPUT_ROOT as DATA_ROOT

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
                 upgrade: np.array, sale: pd.DataFrame, data_root=DATA_ROOT):
        """
        Methods
        --------
        `validate_shape`
        `validate_capacity_obj`
        `validate_obj`
        """
        self.scenario = scenario
        self.instance_id = instance_id
        self.acceptance = acceptance
        self.upgrade = upgrade
        reader = CSVDataReader(scenario, data_root)
        self.order_price, self.order_room, self.order_stay = \
            reader.collect_agent_info(instance_id)
        self.capacity, self.upgrade_fee = reader.collect_hotel_info(instance_id)
        self.ind_pmf, self.ind_room_price, self.demand_ub = \
            reader.collect_individual_info()
        self.error_base = ConstraintViolation(self.scenario, self.instance_id)
        self.num_room = self.capacity.shape[0]
        self.num_order = self.order_price.shape[0]
        self.num_period = self.order_stay.shape[1]
        self.sale = sale
        
    def validate_shape(self, rule):
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

        if rule == "up":
            if (np.tril(self.upgrade) > 0).any():
                self.error_base.hint = "Nonzero in lower triangle of upgrade"
                raise self.error_base
        elif rule == "down":
            if (np.triu(self.upgrade) > 0).any():
                self.error_base.hint = "Nonzero in upper triangle of upgrade"
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
        base_capacity = np.repeat(
            np.reshape(self.capacity, (self.num_room, 1)), 
            self.num_period,
            axis=1
        )
        invalid_index = (final_demand > base_capacity)
        if invalid_index.any():
            if not (np.isclose(
                final_demand[invalid_index], base_capacity[invalid_index],
                rtol=0, atol=1
            ).all()):
                self.error_base.hint = "Demand exceeds capacity"
                raise self.error_base
        self.vacancy = (self.capacity.reshape((self.num_room, 1)) * 
                        np.ones(self.num_period))
        self.vacancy = self.vacancy - final_demand
        self.agent_demand = final_demand

    def validate_capacity_obj(self, obj):
        self._validate_capacity()
        
        agent_rev = np.dot(
            self.acceptance, 
            self.order_price.reshape((self.num_order, 1))
        )[0]
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
        # For the calculation convenience, the matrix has the same max demand_ub 
        # elements for the last dimension. However, the PMF would be zero, so 
        # the product result should be guaranteed.
        sale = np.amin(np.concatenate([vacancy, ind_quantity], axis=3), axis=3)
        mul_index = pd.MultiIndex.from_product(
            [[str(r + 1) for r in range(self.num_room)], 
             [str(t + 1) for t in range(self.num_period)],
             [str(o + 1) for o in range(self.ind_pmf.shape[2])]],
            names=["room", "time", "outcome"]
        )
        sale_df = pd.DataFrame(sale.flatten(), index=mul_index, columns=["sale"])
        sale_merge = sale_df.merge(self.sale, left_index=True, right_index=True, 
                                suffixes=['_validator', '_gurobi'], how='left', 
                                indicator=True)
        # sale_merge.to_csv("merge.csv")
        sale_gurobi = sale_merge['sale_gurobi'].fillna(0).to_numpy().reshape(
            (self.num_room, self.num_period, self.ind_pmf.shape[2])
        )

        ind_rev = (sale * self.ind_pmf * 
                   self.ind_room_price.reshape((self.num_room, 1, 1)))
        total_exp_rev = agent_rev + ind_rev.sum()

        if not np.isclose(
            (sale_gurobi * self.ind_pmf * 
             self.ind_room_price.reshape((self.num_room, 1, 1))).sum(), 
            ind_rev.sum(), 
            rtol=0, 
            atol=1
        ):
            # TODO would test with any or all not the sum product 
            # to more thoroughly
            # FIXME magic number around np.isclose
            self.error_base.hint = "Effective sale not match"
            raise self.error_base

        if not np.isclose(total_exp_rev, obj, rtol=0, atol=1):
            # FIXME magic number around np.isclose
            self.error_base.hint = "Obj error"
            raise self.error_base 
        

