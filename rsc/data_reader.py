import json
import pandas as pd
import numpy as np
import xarray as xr
from os.path import join

DATA_ROOT = join("data")

class JSONDataReader:
    """Help yulindog hide some stuff"""
    def __init__(self, scenario: dict, data_root=DATA_ROOT) -> None:
        self.scenario = scenario
        self.data_root = data_root

    def collect_agent_info(self, instance_id):
        """
        Returns
        -------
        agent_order_set : list
        time_span : set
        agent_order_price: dict
        agent_order_room_quantity : dict
        agent_order_stay: dict
        """
        with open(
            join(self.data_root, "agent_order_price", self.scenario["agent"], f"{instance_id}.json")
        ) as f:
            agent_order_price = json.load(f)
        with open(join(self.data_root, "agent_order_room_quantity", self.scenario["agent"],
                       f"{instance_id}.json")) as f:
            agent_order_room_quantity = json.load(f)
        with open(
            join(self.data_root, "agent_order_stay", self.scenario["agent"], f"{instance_id}.json")
        ) as f:
            agent_order_stay = json.load(f)
        time_span = list(set(
            period
            for stay_set in agent_order_stay.values() for period in stay_set
        ))
        time_span.sort()
        time_span = [str(period) for period in time_span]
        agent_order_set = list(agent_order_price.keys())
        # time_span is defined by order. 
        return (agent_order_set, time_span, agent_order_price,
                agent_order_room_quantity, agent_order_stay)

    def collect_hotel_info(self, upgrade_rule: str):
        """
        Parameters
        -----------
        upgrade_rule: `up`, `down` or `both`
        Returns
        -------
        room_type_set : list
        upgrade_levels : dict
        downgrade_levels : dict
        room_capacity : dict
        upgrade_fee : dict
        """
        with open(
            join(self.data_root, f"capacity.json")
        ) as f:
            room_capacity = json.load(f)
        with open(
            join(self.data_root, "upgrade_fee.json")
        ) as f:
            upgrade_fee = json.load(f)

        room_type_set = list(room_capacity.keys())
        up_levels = {
            i: [j for j in room_type_set if int(j) > int(i)]
            for i in room_type_set
        }
        down_levels = {
            i: [j for j in room_type_set if int(j) < int(i)]
            for i in room_type_set
        }
        if upgrade_rule == "up":
            upgrade_levels = up_levels
            downgrade_levels = down_levels
        elif upgrade_rule == "down":
            upgrade_levels = down_levels
            downgrade_levels = up_levels
        elif upgrade_rule == "both":
            upgrade_levels = {
                i: [j for j in room_type_set if int(j) != int(i)]
                for i in room_type_set
            }
            downgrade_levels = upgrade_levels
        
        return (room_type_set, upgrade_levels, downgrade_levels, room_capacity,
                upgrade_fee)

    def collect_individual_info(self):
        """
        Returns
        -------
        individual_demand_pmf : dict
        individual_room_price : dict
        """
        with open(
            join(self.data_root, "individual_room_price.json")
        ) as f:
            individual_room_price = json.load(f)
        with open(
            join(self.data_root, "individual_demand_pmf", f'{self.scenario["individual"]}.json')
        ) as f:
            individual_demand_pmf = json.load(f)
        with open(
            join(self.data_root, "demand_ub.json")
        ) as f:
            demand_ub = json.load(f)
        return individual_demand_pmf, individual_room_price, demand_ub

# FIXME NOT compatible NOW
class CSVDataReader:
    """Help yulindog hide some stuff"""
    def __init__(self, scenario: dict, data_root=DATA_ROOT) -> None:
        self.scenario = scenario
        self.data_root = data_root

    def collect_agent_info(self, instance_id):
        """
        Returns
        -------
        agent_order_price: 1-D array
        agent_order_room_quantity : 2-D array
        agent_order_stay: 2-D array
        """
        agent_order_price = pd.read_csv(
            join(self.data_root, "agent_order_price", self.scenario["agent"], 
                 f"{instance_id}.csv")
        )['value'].to_numpy()
        agent_order_room_quantity = pd.read_csv(
            join(self.data_root, "agent_order_room_quantity", 
                 self.scenario["agent"], f"{instance_id}.csv")
        ).to_numpy()
        agent_order_stay = pd.read_csv(
            join(self.data_root, "agent_order_stay", self.scenario["agent"], 
                 f"{instance_id}.csv")
        ).to_numpy()
        return agent_order_price, agent_order_room_quantity, agent_order_stay

    def collect_hotel_info(self, instance_id):
        """
        Warning
        --------
        Upper triangular mask for upgrade is missing.
        Returns
        -------
        room_capacity : 1-D array
        upgrade_fee : 2-D array
        """
        room_capacity = pd.read_csv(
            join(self.data_root, "capacity.csv")
        )['value'].to_numpy()
        upgrade_fee = pd.read_csv(
            join(self.data_root, "upgrade_fee.csv")
        ).to_numpy()

        return (room_capacity, upgrade_fee)

    def collect_individual_info(self):
        """
        Returns
        -------
        individual_demand_prob : 3-D array
        individual_room_price : 1-D array
        """
        individual_room_price = pd.read_csv(
            join(self.data_root, "individual_room_price.csv")
        )['value'].to_numpy()
        ds = xr.open_dataset(
            join(self.data_root, "individual_demand_prob", 
                 f'{self.scenario["individual"]}.nc')
        )
        individual_demand_prob = ds.to_array()[0].values
        # individual_demand_prob = ds.to_dataframe().to_numpy().reshape(
        #     [ds.dims[key] for key in ds.dims]
        # )
        demand_ub = pd.read_csv(
            join(self.data_root, "demand_ub.csv")
        ).to_numpy()
        return individual_demand_prob, individual_room_price, demand_ub


# folder = join(DATA_ROOT, scenario)
# given one instance

# reader = CSVDataReader({"agent": "stay_mul_0.048_high_request_room_id_0", "individual": "ind_demand_0.5"})
# reader.collect_individual_info()