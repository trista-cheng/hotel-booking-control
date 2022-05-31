import numpy as np
import pandas as pd
import logging
import configparser

from os.path import join
from pathlib import Path
from time import perf_counter

from gurobi_optimizer_mix import GurobiManager
from gurobi_optimizer_relax import GurobiRelaxManager
from tools import get_exp_ub
# from metric import get_reject_room_ratio


logging.basicConfig(filename='log.log',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.WARNING, datefmt='%Y-%m-%d %H:%M:%S')
logging.warning("Start!")

# important settings

DATA_ROOT = "data"
INSTANCE_NUM = 5
MIP_GAP = 0.1
UPGRADE_RULE = "up"
SET_ORDER_ACC = False
RELAX = True
SOLUTION_DIR = f"solutions/{UPGRADE_RULE}_algo_{SET_ORDER_ACC}_relax_{RELAX}"

# TODO consider to put index information in scenarios.ini
def get_index(scenario):
    room_rate = np.array(scenario['room_rate'][1:-1].split(), dtype=float)
    hot_room = room_rate.argmax()
    return [float(scenario['stay_mul']), hot_room, float(scenario['ind_demand_mul'])]

scenarios = configparser.ConfigParser()
scenarios.read('scenarios.ini')

# lacks = []
objs = []
times = []
index = []
with_ind_cancel = True
# TODO nested for-loop is birdy~ use class to hide it
for with_capacity_reservation in [False, True, ]:
    for with_agent_cancel in [False, True, ]:
        for scenario_name in scenarios.sections():
            scenario = scenarios[scenario_name]
            index.append(
                [with_ind_cancel, with_agent_cancel, with_capacity_reservation]
                + get_index(scenario)
            )

            # lack = []
            obj = []
            cal_time = []
            ub = []
            for instance_id in range(INSTANCE_NUM):
                start_time = perf_counter()
                if RELAX:
                    optimizer = GurobiRelaxManager(
                        scenario,
                        instance_id,
                        UPGRADE_RULE,
                        with_capacity_reservation=with_capacity_reservation,
                        with_ind_cancel=with_ind_cancel,
                        with_agent_cancel=with_agent_cancel,
                        set_order_acc=SET_ORDER_ACC,
                        relax=RELAX
                    )
                else:
                    optimizer = GurobiManager(
                        scenario,
                        instance_id,
                        UPGRADE_RULE,
                        with_capacity_reservation=with_capacity_reservation,
                        with_ind_cancel=with_ind_cancel,
                        with_agent_cancel=with_agent_cancel,
                        set_order_acc=SET_ORDER_ACC
                    )
                # acceptance: 1 x order
                # upgrade: order x room x room
                optimizer.build_model()
                optimizer.solve(time_limit=float('inf'), mip_gap=MIP_GAP)
                (acceptance_df, upgrade_df, ind_valid_df, comp_df, rev_df,
                 obj_val) = optimizer.get_result()

                # test = Validator(scenario, instance_id, acceptance, upgrade, sale)
                # try:
                #     test.validate_shape(rule=UPGRADE_RULE)
                #     test.validate_capacity_obj(obj_val)
                # except:
                #     continue

                output_folder = join(
                    SOLUTION_DIR,
                    (f'agent_{int(with_agent_cancel)}'
                     f'_res_{int(with_capacity_reservation)}'),
                    scenario_name
                )
                Path(output_folder).mkdir(exist_ok=True, parents=True)
                acceptance_df.to_csv(
                    join(output_folder, f"{instance_id}_acceptance.csv")
                )
                upgrade_df.to_csv(
                    join(output_folder, f"{instance_id}_upgrade.csv"),
                )
                ind_valid_df.to_csv(
                    join(output_folder, f"{instance_id}_ind_valid.csv"),
                )
                comp_df.to_csv(
                    join(output_folder, f"{instance_id}_comp.csv"),
                )
                rev_df.to_csv(
                    join(output_folder, f'{instance_id}_rev.csv')
                )
                # lack_value = get_reject_room_ratio(scenario, instance_id,
                #                                     acceptance, data_root=DATA_ROOT)
                # lack.append(lack_value)
                obj.append(obj_val)
                cal_time.append(perf_counter() - start_time)
                logging.warning(f"agent {with_agent_cancel}, "
                                f"res {with_capacity_reservation} "
                                f"{scenario_name}, {instance_id}")
                agent_ub, ind_ub, capacity_value = get_exp_ub(
                    scenario, instance_id, UPGRADE_RULE, with_agent_cancel
                )
                ub.append([agent_ub, ind_ub, capacity_value])
            scenario_result = pd.DataFrame({"time": cal_time, "obj": obj})
            scenario_result[['agent_ub', 'ind_ub', 'capacity_value']] = ub
            scenario_result.to_csv(
                join(output_folder, "time_obj.csv"), index=False
            )
            # lacks.append(np.mean(lack))
            objs.append(np.mean(obj))
            times.append(np.mean(cal_time))

result = np.hstack([
    np.array(objs).reshape((-1, 1)),
    # np.array(lacks).reshape((-1, 1)),
    np.array(times).reshape((-1, 1))
])
index = pd.MultiIndex.from_tuples(
    index,
    names=["ind cancel", "agent order cancel", "capacity reservation",
           "stay mul", "high request room", "ind demand mul"]
)
pd.DataFrame(
    result,
    index=index,
    columns=["obj", "time"]  # "reject capacity ratio",
).to_csv(f"avg_result__{UPGRADE_RULE}__algo_{SET_ORDER_ACC}__relax_{RELAX}.csv")
logging.warning("End!")
