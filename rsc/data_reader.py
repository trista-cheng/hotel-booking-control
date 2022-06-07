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

    def collect_agent_info(self, instance_id, factor_key='agent_factor'):
        """
        Returns
        -------
        agent_order_set : list
        time_span : set
        agent_order_price: dict
        agent_order_room_quantity : dict
        agent_order_stay: dict
        agent_cancel_rate: dict
        """
        with open(
            join(self.data_root, "agent_order_price",
                 self.scenario[factor_key], f"{instance_id}.json")
        ) as f:
            agent_order_price = json.load(f)
        with open(join(self.data_root, "agent_order_room_quantity",
                       self.scenario[factor_key], f"{instance_id}.json")) as f:
            agent_order_room_quantity = json.load(f)
        with open(
            join(self.data_root, "agent_order_stay", self.scenario[factor_key],
                 f"{instance_id}.json")
        ) as f:
            agent_order_stay = json.load(f)
        with open(
            join(self.data_root, "agent_cancel_rate", self.scenario[factor_key],
                 f"{instance_id}.json")
        ) as f:
            agent_cancel_rate = json.load(f)

        time_span = list(set(
            period
            for stay_set in agent_order_stay.values() for period in stay_set
        ))
        time_span.sort()
        time_span = [str(period) for period in time_span]
        agent_order_set = list(agent_order_price.keys())
        # time_span is defined by order.
        return (agent_order_set, time_span, agent_order_price,
                agent_order_room_quantity, agent_order_stay, agent_cancel_rate)

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
        compensation_price: dict
        """
        with open(
            join(self.data_root, f"capacity.json")
        ) as f:
            room_capacity = json.load(f)
        with open(
            join(self.data_root, "upgrade_fee.json")
        ) as f:
            upgrade_fee = json.load(f)
        with open(join(self.data_root, "compensation_price.json")) as f:
            compensation_price = json.load(f)

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
        elif upgrade_rule == "no":
            empty_levels = {i: []for i in room_type_set}
            upgrade_levels = empty_levels
            downgrade_levels = empty_levels

        return (room_type_set, upgrade_levels, downgrade_levels, room_capacity,
                upgrade_fee, compensation_price)

    def collect_individual_info(self, factor_key='individual_factor'):
        """
        Returns
        -------
        individual_demand_pmf : dict
        individual_room_price : dict
        demand_ub: dict
        demand_ub: dict
        """
        with open(join(self.data_root, "individual_room_price.json")) as f:
            individual_room_price = json.load(f)
        with open(
            join(self.data_root, "individual_demand_pmf",
                 f'{self.scenario[factor_key]}.json')
        ) as f:
            individual_demand_pmf = json.load(f)
        with open(join(self.data_root, "demand_ub.json")) as f:
            demand_ub = json.load(f)
        with open(join(self.data_root, "individual_cancel_rate.json")) as f:
            individual_cancel_rate = json.load(f)
        return individual_demand_pmf, individual_room_price, demand_ub, \
               individual_cancel_rate

# FIXME NOT compatible NOW
class CSVDataReader:
    """Help yulindog hide some stuff"""
    def __init__(self, scenario: dict, data_root=DATA_ROOT) -> None:
        self.scenario = scenario
        self.data_root = data_root

    def collect_agent_info(self, instance_id, factor_key='agent_factor'):
        """
        Returns
        -------
        agent_order_set : list
        time_span : list
        agent_order_price: list
        agent_order_room_quantity: 2D array
        agent_order_stay: 2D array
        agent_cancel_rate: list
        """
        agent_order_price = pd.read_csv(
            join(self.data_root, "agent_order_price",
                 self.scenario[factor_key], f"{instance_id}.csv")
        ).to_numpy().flatten()
        agent_order_room_quantity = pd.read_csv(
            join(self.data_root, "agent_order_room_quantity",
                 self.scenario[factor_key], f"{instance_id}.csv")
        ).to_numpy()
        agent_order_stay = pd.read_csv(
            join(self.data_root, "agent_order_stay", self.scenario[factor_key],
                 f"{instance_id}.csv")
        ).to_numpy()
        agent_cancel_rate = pd.read_csv(
            join(self.data_root, "agent_cancel_rate", self.scenario[factor_key],
                 f"{instance_id}.csv")
        ).to_numpy().flatten()

        time_span = np.arange(agent_order_stay.shape[1])
        agent_order_set = np.arange(len(agent_order_price))
        # time_span is defined by order.
        return (agent_order_set, time_span, agent_order_price,
                agent_order_room_quantity, agent_order_stay, agent_cancel_rate)

    def collect_hotel_info(self, upgrade_rule: str):
        """
        Parameters
        -----------
        upgrade_rule: `up`, `down` or `both`
        Returns
        -------
        room_type_set : 1D array
        room_capacity : 1D array
        upgrade_fee : 2D array
        compensation_price: 2D array
        """
        room_capacity = pd.read_csv(
            join(self.data_root, f"capacity.csv")
        ).to_numpy().flatten()
        upgrade_fee = pd.read_csv(
            join(self.data_root, "upgrade_fee.csv")
        ).to_numpy()
        compensation_price = pd.read_csv(
            join(self.data_root, "compensation_price.csv")
        ).to_numpy()

        room_type_set = np.arange(len(room_capacity))
        if upgrade_rule == 'up':
            invalid_msk = np.tril_indices_from(upgrade_fee)
            upgrade_fee[invalid_msk] = 0
        elif upgrade_rule == 'down':
            invalid_msk = np.triu_indices_from(upgrade_fee)
            upgrade_fee[invalid_msk] = 0

        return (room_type_set, room_capacity, upgrade_fee, compensation_price)

    def collect_individual_info(self, factor_key='individual_factor'):
        """
        Returns
        -------
        ind_demand_prob : 3D numpy (room x time x quant)
        individual_room_price : 1D array (room)
        demand_ub: 1D array (room)
        individual_cancel_rate : 1D array
        """
        individual_room_price = pd.read_csv(
            join(self.data_root, "individual_room_price.csv")
        ).to_numpy().flatten()

        with open(
            join(self.data_root, "individual_demand_pmf",
                 f'{self.scenario[factor_key]}.npy'),
            'rb'
        ) as f:
            individual_demand_pmf = np.load(f, allow_pickle=True)
        ind_demand_prob = pd.DataFrame.from_dict(
            individual_demand_pmf[()], orient='index'
        ).sort_index()
        # the index is int not str, otherwise sort may result in inconsistancy
        # FIXME not sure the numpy object is correct
        ind_demand_prob = ind_demand_prob['prob'].to_numpy().reshape(
            ind_demand_prob.index.levshape
        )
        demand_ub = pd.read_csv(
            join(self.data_root, "demand_ub.csv")
        ).to_numpy().flatten()
        individual_cancel_rate = pd.read_csv(
            join(self.data_root, "individual_cancel_rate.csv")
        ).to_numpy().flatten()
        return (ind_demand_prob, individual_room_price, demand_ub,
                individual_cancel_rate)


# folder = join(DATA_ROOT, scenario)
# given one instance

# reader = CSVDataReader({"agent": "stay_mul_0.048_high_request_room_id_0", "individual": "ind_demand_0.5"})
# reader.collect_individual_info()