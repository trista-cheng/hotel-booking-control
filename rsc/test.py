import configparser
import pandas as pd
import numpy as np

from os.path import join
from itertools import product

from solver import Solver

# warning for some restrictions
WITH_IND_CANCEL = True  # WARNING! it is only valid as it is True.
RELAX = False  # SHOULD ALWAYS be False,
# since relacx optimizer is not maintained
SET_ORDER_ACC = False  # only use either gurobi or solver not partial

# important settings
REPLICATE_NUM = 30
# ROOT = join('history', '0606_too_large')
ROOT = ''
DATA_ROOT = join(ROOT, 'data')
UPGRADE_RULE = "up"

# test factor
SOLVERS = ['gurobi']
CAP_REV_LEVLES = [0, ]
AGENT_CANCEL_LEVELS = [0, ]
SCENARIOS = configparser.ConfigParser()
SCENARIOS.read(join(ROOT, 'scenarios.ini'))

NUM_TYPE = 4
TIME_LEN = 14

for solver in SOLVERS:
    algo_on = 1 if solver == 'yulindog' else 0

    # for different environments
    for with_capacity_reservation, with_agent_cancel in \
        product(CAP_REV_LEVLES, AGENT_CANCEL_LEVELS):
        solution_dir = (f"solution_{UPGRADE_RULE}/algo_{algo_on}/"
                        f"overbooking_{with_capacity_reservation}__"
                        f"agent_cancel_{with_agent_cancel}")
        solution_dir = join(ROOT, solution_dir)

        # for each scenario
        # FIXME modify here is birdy
        for scenario_name in SCENARIOS.sections():
            print(scenario_name)
            scenario = SCENARIOS[scenario_name]
            output_folder = join(solution_dir, scenario_name)
            for instance_id in range(REPLICATE_NUM):
                acceptance_df = pd.read_csv(
                    join(output_folder, f"{instance_id}_acceptance.csv"),
                    index_col=0
                )
                upgrade_df = pd.read_csv(
                    join(output_folder, f"{instance_id}_upgrade.csv"),
                    index_col=[0, 1, 2]
                )
                cap_rev_df = pd.read_csv(
                    join(output_folder, f"{instance_id}_cap_rev.csv"),
                    index_col=0,
                    skiprows=(1, 2)
                )
                order_acceptance = acceptance_df.to_numpy().flatten()
                order_upgrade = np.zeros(
                    (len(order_acceptance), NUM_TYPE, NUM_TYPE,)
                )
                for index, value in upgrade_df.iterrows():
                    order_upgrade[index[0] - 1, index[1] - 1, index[2] - 1] =  \
                        value['upgrade amount']
                capacity_reservation = cap_rev_df.to_numpy()

                solver = Solver(
                    scenario,
                    instance_id,
                    UPGRADE_RULE,
                    with_agent_cancel=with_agent_cancel,
                    with_capacity_reservation=with_capacity_reservation,
                    with_ind_cancel=WITH_IND_CANCEL,
                    data_root=DATA_ROOT
                )
                obj_val = solver.get_obj(order_acceptance, order_upgrade,
                                         capacity_reservation)
                gurobi_obj = pd.read_csv(join(output_folder, 'performance.csv'),
                                         index_col=0)
                g_obj = gurobi_obj.loc[instance_id, 'obj']
                print(obj_val-g_obj)
