import json
import pandas as pd
import numpy as np
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
        # FIXME a key is nonsense
        with open(
            join(self.data_root, "agent_order_price", self.scenario["a"], f"{instance_id}.json")
        ) as f:
            agent_order_price = json.load(f)
        with open(join(self.data_root, "agent_order_room_quantity", self.scenario["a"],
                       f"{instance_id}.json")) as f:
            agent_order_room_quantity = json.load(f)
        with open(
            join(self.data_root, "agent_order_stay", self.scenario["a"], f"{instance_id}.json")
        ) as f:
            agent_order_stay = json.load(f)
        time_span = list(set(
            period
            for stay_set in agent_order_stay.values() for period in stay_set
        ))
        time_span.sort()
        time_span = [str(period) for period in time_span]
        agent_order_set = list(agent_order_price.keys())
        return (agent_order_set, time_span, agent_order_price,
                agent_order_room_quantity, agent_order_stay)

    def collect_hotel_info(self, instance_id):
        """
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
            join(self.data_root, "upgrade_fee", self.scenario["a"], f"{instance_id}.json")
        ) as f:
            upgrade_fee = json.load(f)

        room_type_set = list(room_capacity.keys())
        upgrade_levels = {
            i: [j for j in room_type_set if int(j) > int(i)]
            for i in room_type_set
        }
        downgrade_levels = {
            i: [j for j in room_type_set if int(j) < int(i)]
            for i in room_type_set
        }
        return (room_type_set, upgrade_levels, downgrade_levels, room_capacity,
                upgrade_fee)

    def collect_individual_info(self, instance_id):
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
            join(self.data_root, "individual_demand_pmf", self.scenario["i"], f"{instance_id}.json")
        ) as f:
            individual_demand_pmf = json.load(f)
        return individual_demand_pmf, individual_room_price


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
        # FIXME should incorporate with factor
        agent_order_price = pd.read_csv(
            join(self.data_root, "agent_order_price", f"{instance_id}.csv")
        )['value'].to_numpy()
        agent_order_room_quantity = pd.read_csv(
            join(self.data_root, "agent_order_room_quantity",
                 f"{instance_id}.csv")
        ).to_numpy()
        agent_order_stay = pd.read_csv(
            join(self.data_root, "agent_order_stay", f"{instance_id}.csv")
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
            join(self.data_root, "room_capacity", f"{instance_id}.csv")
        )['value'].to_numpy()
        upgrade_fee = pd.read_csv(
            join(self.data_root, "upgrade_fee", f"{instance_id}.csv")
        ).to_numpy()

        return (room_capacity, upgrade_fee)

    def collect_individual_info(self, instance_id):
        """
        Returns
        -------
        individual_demand_prob : 2-D array
        individual_demand_quantity : 2-D array
        individual_room_price : 1-D array
        """
        individual_room_price = pd.read_csv(
            join(self.data_root, "individual_room_price", f"{instance_id}.csv")
        )['value'].to_numpy()
        individual_demand_quantity = pd.read_csv(
            join(self.data_root, "individual_demand_quantity",
                 f"{instance_id}.csv"))
        for col in individual_demand_quantity.columns:
            individual_demand_quantity[col] = \
                individual_demand_quantity[col].apply(
                lambda x:  np.array(x[1: -1].split('", "'), dtype=int))
        individual_demand_prob = pd.read_csv(
            join(self.data_root, "individual_demand_prob",
                 f"{instance_id}.csv"))
        for col in individual_demand_prob.columns:
            individual_demand_prob[col] = individual_demand_prob[col].apply(
                lambda x:  np.array(x[1: -1].split('", "'), dtype=float))
        individual_demand_quantity = individual_demand_quantity.to_numpy()
        individual_demand_prob = individual_demand_prob.to_numpy()
        return individual_demand_prob, individual_demand_quantity, \
            individual_room_price


# folder = join(DATA_ROOT, scenario)
# given one instance
