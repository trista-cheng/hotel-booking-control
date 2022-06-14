import configparser
import pandas as pd

from os.path import join
from itertools import product
from pathlib import Path

UPGRADE_RULE = 'up'

ROOT = ''
REPORT_DIR = 'report'
SOLVERS = ['gurobi', 'yulindog']
CAP_REV_LEVLES = [0]
AGENT_CANCEL_LEVELS = [0, 1]
SCENARIOS = configparser.ConfigParser()
SCENARIOS.read(join(ROOT, 'scenarios.ini'))
ADVANCED = True

# multi_index = pd.MultiIndex.from_product([CAP_REV_LEVLES, AGENT_CANCEL_LEVELS,
#                                           SCENARIOS.sections()])
summary = pd.DataFrame()
# for different environments
for with_capacity_reservation, with_agent_cancel in \
    product(CAP_REV_LEVLES, AGENT_CANCEL_LEVELS):
    gurobi_dir = join(ROOT, f"solution_{UPGRADE_RULE}", 'algo_0')
    algo_dir = join(f"solution_{UPGRADE_RULE}", 'algo_1')

    gurobi_dir = join(gurobi_dir, (f"overbooking_{with_capacity_reservation}__"
                      f"agent_cancel_{with_agent_cancel}"))
    algo_dir = join(algo_dir, (f"overbooking_{with_capacity_reservation}__"
                    f"agent_cancel_{with_agent_cancel}"))

    # for each scenario
    # for each scenario
    # FIXME modify here is birdy
    for scenario_name in SCENARIOS.sections():
        scenario = SCENARIOS[scenario_name]
        gurobi_performance = pd.read_csv(
            join(gurobi_dir, scenario_name, 'performance.csv'),
            index_col=0
        )
        algo_performance = pd.read_csv(
            join(algo_dir, scenario_name, 'performance.csv'),
            index_col=0
        )
        comparison = pd.concat(
            [gurobi_performance.add_prefix('gurobi_'),
             algo_performance.iloc[:, :2].add_prefix('algo_')],
            axis=1
        )
        comparison['est_optimal'] = (
            comparison['gurobi_obj'] *
            1 / (1 - comparison['gurobi_mip_gap'])
        )
        comparison[['algo_obj_ratio_by_gurobi', 'algo_time_ratio_by_gurobi']] = \
            (algo_performance.iloc[:, :2] / gurobi_performance.iloc[:,:2])
        comparison['algo_obj_ratio_by_est'] = (
            algo_performance['obj'] / comparison['est_optimal']
        )

        if ADVANCED:
            advanced_algo_dir = join(f"solution_{UPGRADE_RULE}", 'algo_1')
            advanced_algo_dir = join(advanced_algo_dir,
                                     (f"overbooking_1__"
                                      f"agent_cancel_{with_agent_cancel}"))
            advanced_algo = pd.read_csv(
                join(advanced_algo_dir, scenario_name, 'performance.csv'),
                index_col=0
            )
            comparison = pd.concat(
                [comparison,
                advanced_algo.iloc[:, :2].add_prefix('algo_overbooking_')],
                axis=1
            )
            comparison[['algo_overbooking_obj_ratio_by_gurobi',
                        'algo_overbooking_time_ratio_by_gurobi']] = \
                (advanced_algo.iloc[:, :2] / gurobi_performance.iloc[:,:2])
            comparison['algo_overbooking_obj_ratio_by_est'] = (
                advanced_algo['obj'] / comparison['est_optimal']
            )

        folder = join(REPORT_DIR, UPGRADE_RULE,
                      (f"overbooking_{with_capacity_reservation}__"
                       f"agent_cancel_{with_agent_cancel}"))
        Path(folder).mkdir(exist_ok=True, parents=True)
        comparison.to_csv(join(folder, f"{scenario_name}.csv"), index=False)
        result = comparison.mean().to_frame().T
        result[['overbooking', 'agent_cancel']] = [with_capacity_reservation,
                                                   with_agent_cancel]
        result[['stay_mul', 'top_room_rate', 'ind_demand_mul']] = [
            scenario['stay_mul'], scenario['room_rate_name'],
            scenario['ind_demand_mul']
        ]
        summary = pd.concat([summary, result], axis=0)

summary.to_csv(join(REPORT_DIR, 'summary.csv'), index=False)
