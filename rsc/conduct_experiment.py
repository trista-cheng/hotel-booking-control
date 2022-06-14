import numpy as np
import pandas as pd
import logging
import configparser
# import atexit

from os.path import join
from pathlib import Path
from time import perf_counter
from itertools import product

from gurobi_optimizer_mix import GurobiManager
from solver import Solver
from tools import get_exp_ub

logging.basicConfig(filename='algo_log.log',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.WARNING, datefmt='%Y-%m-%d %H:%M:%S')
logging.warning("Start!")

# warning for some restrictions
WITH_IND_CANCEL = True  # WARNING! it is only valid as it is True.
RELAX = False  # SHOULD ALWAYS be False,
# since relacx optimizer is not maintained
SET_ORDER_ACC = False  # only use either gurobi or solver not partial

# important settings
REPLICATE_NUM = 15
MIP_GAP = 0.05
TIME_LIMIT = 3 * 60 * 60
# ROOT = join("history", "0609_small")
ROOT = ''
DATA_ROOT = join(ROOT, "data")
UPGRADE_RULE = "up"

# test factor
SOLVERS = ['gurobi']
CAP_REV_LEVLES = [0, ]
AGENT_CANCEL_LEVELS = [0, 1]
SCENARIOS = configparser.ConfigParser()
SCENARIOS.read(join(ROOT, 'scenarios.ini'))

# TODO consider to put index information in scenarios.ini
# not used
def get_index(scenario):
    room_rate = np.array(scenario['room_rate'][1:-1].split(), dtype=float)
    hot_room = room_rate.argmax()
    return ([float(scenario['stay_mul']), hot_room,
             float(scenario['ind_demand_mul'])])

class Conductor:
    def __init__(self, solvers: list, cap_rev_levels: list,
                 agent_cancel_levels: list, scenarios: configparser,
                 replicate_num: int, upgrade_rule: str, data_root: str):
        """Conduct full trials with given environments by solvers

        Args:
            solvers (list): `['gurobi', 'yulindog']`
            cap_rev_levels (list): `[0, 1]`
            agent_cancel_levels (list): `[0, 1]`
            scenarios (configparser)
            replicate_num (int)
            upgrade_rule (str): options are `up`, `down` and `both`.
            data_root (str): set by params.
        """
        self.cap_rev_levels = cap_rev_levels
        self.agent_cancel_levels = agent_cancel_levels
        self.scenarios = scenarios
        self.solvers = solvers
        self.replicate_num = replicate_num
        self.upgrade_rule = upgrade_rule
        self.data_root = data_root

    def save_scenario_metric(self, output_folder, objs, cal_times, mip_gaps,
                             ubs):
        df = pd.DataFrame({"obj": objs, "time": cal_times, 'mip_gap': mip_gaps})
        df[['agent_ub', 'ind_ub', 'capacity_value', 'income_ub']] = ubs
        df.to_csv(join(output_folder, 'performance.csv'))

    def conduct_experiment(self):
        # for different solvers
        for solver in self.solvers:
            algo_on = 1 if solver == 'yulindog' else 0

            # for different environments
            for with_capacity_reservation, with_agent_cancel in \
                product(self.cap_rev_levels, self.agent_cancel_levels):
                solution_dir = (f"solution_{self.upgrade_rule}/algo_{algo_on}/"
                                f"overbooking_{with_capacity_reservation}__"
                                f"agent_cancel_{with_agent_cancel}")

                # for each scenario
                # FIXME modification is birdy
                for scenario_name in self.scenarios.sections():
                    scenario = self.scenarios[scenario_name]
                    output_folder = join(solution_dir, scenario_name)
                    Path(output_folder).mkdir(exist_ok=True, parents=True)
                    objs = []
                    cal_times = []
                    ubs = []
                    mip_gaps= []

                    for instance_id in range(self.replicate_num):
                        experiment = Experiment(
                            solver=solver,
                            with_capacity_reservation=with_capacity_reservation,
                            with_agent_cancel=with_agent_cancel,
                            scenario=scenario,
                            instance_id=instance_id,
                            output_folder=output_folder,
                            upgrade_rule=self.upgrade_rule,
                            data_root=self.data_root,
                        )
                        obj, cal_time, mip_gap, ub = experiment.carry_out_trial()
                        objs.append(obj)
                        cal_times.append(cal_time)
                        mip_gaps.append(mip_gap)
                        ubs.append(ub)

                    self.save_scenario_metric(output_folder, objs, cal_times,
                                              mip_gaps, ubs)



class Experiment:
    def __init__(self, solver: str, with_capacity_reservation: bool,
                 with_agent_cancel: bool, scenario: dict, instance_id: int,
                 output_folder: str, upgrade_rule: str, data_root: str) -> None:
        """Do one trial and return metrics

        Args:
            solver (str): `gurobi` or `yulindog`
            with_capacity_reservation (bool): _description_
            with_agent_cancel (bool): _description_
            scenario (dict): _description_
            instance_id (int)
            output_folder (str)
            upgrade_rule
            data_root
        """
        self.solver = solver
        self.with_capacity_reservation = with_capacity_reservation
        self.with_agent_cancel = with_agent_cancel
        self.scenario = scenario
        self.instance_id = instance_id
        self.upgrade_rule = upgrade_rule
        self.data_root = data_root
        self.output_folder = output_folder

    def save_instance_sol(self, acceptance_df, upgrade_df, cap_rev_df,
                          obj_val, cal_time, mip_gap):
        acceptance_df.to_csv(
            join(self.output_folder, f"{self.instance_id}_acceptance.csv"),
            # index=False
        )
        upgrade_df.to_csv(
            join(self.output_folder, f"{self.instance_id}_upgrade.csv"),
            # index=False
        )
        cap_rev_df.to_csv(
            join(self.output_folder, f"{self.instance_id}_cap_rev.csv"),
            # index=False
        )
        pd.DataFrame(
            {'obj': [obj_val], 'time': [cal_time], 'mip_gap': [mip_gap]}
        ).to_csv(
            join(self.output_folder, f"{self.instance_id}_performance.csv"),
            index=False
        )

    def carry_out_trial(self):
        if self.solver == 'gurobi':
            start_time = perf_counter()
            optimizer = GurobiManager(
                self.scenario,
                self.instance_id,
                self.upgrade_rule,
                with_capacity_reservation=self.with_capacity_reservation,
                with_ind_cancel=WITH_IND_CANCEL,
                with_agent_cancel=self.with_agent_cancel,
                set_order_acc=SET_ORDER_ACC
            )
            optimizer.build_model()
            optimizer.solve(time_limit=TIME_LIMIT, mip_gap=MIP_GAP)
            cal_time = perf_counter() - start_time
            (acceptance_df, upgrade_df, cap_rev_df, obj_val, mip_gap,
             ind_valid_df, comp_df, rev_df) = optimizer.get_result()
            # save some extra data for verification
            ind_valid_df.to_csv(join(self.output_folder, f'{self.instance_id}_ind_valid.csv'))
            comp_df.to_csv(join(self.output_folder, f'{self.instance_id}_compensation.csv'))
            rev_df.to_csv(join(self.output_folder, f'{self.instance_id}_revservation.csv'))

        elif self.solver == 'yulindog':
            solver = Solver(
                self.scenario,
                self.instance_id,
                self.upgrade_rule,
                with_agent_cancel=self.with_agent_cancel,
                with_capacity_reservation=self.with_capacity_reservation,
                with_ind_cancel=WITH_IND_CANCEL,
                data_root=DATA_ROOT
            )
            order_acceptance, order_upgrade, capacity_reservation = \
                solver.get_decision()
            # convert to dataframe
            acceptance_df, upgrade_df, cap_rev_df = solver.get_df(
                order_acceptance,
                order_upgrade,
                capacity_reservation
            )
            obj_val = solver.get_obj(order_acceptance, order_upgrade,
                                    capacity_reservation)
            cal_time = solver.calculation_time
            mip_gap = None
        else:
            raise Exception("Error setting in solver name")
        self.save_instance_sol(acceptance_df, upgrade_df, cap_rev_df,
                               obj_val=obj_val, cal_time=cal_time,
                               mip_gap=mip_gap)
        logging.warning(f"solver {self.solver}, "
                        f"overbooking {self.with_capacity_reservation}, "
                        f"agent_cancel {self.with_agent_cancel},"
                        f"{self.scenario['display_name']}, {self.instance_id}")
        agent_ub, ind_ub, capacity_value, income_ub = get_exp_ub(
            self.scenario, self.instance_id, self.upgrade_rule,
            self.with_agent_cancel
        )

        return (obj_val, cal_time, mip_gap,
                [agent_ub, ind_ub, capacity_value, income_ub])


if __name__ == '__main__':

    # def exit_handler():
    #     print('sth wrong')

    # atexit.register(exit_handler)

    conductor = Conductor(SOLVERS, CAP_REV_LEVLES, AGENT_CANCEL_LEVELS,
                          SCENARIOS, REPLICATE_NUM, UPGRADE_RULE, DATA_ROOT)
    conductor.conduct_experiment()
