
from data_reader import CSVDataReader

def get_reject_room_ratio(scenario, instance_id, acceptance):
    reader = CSVDataReader(scenario)
    agent_order_price, order_room_quantity, order_stay = \
        reader.collect_agent_info(instance_id)
    capacity, upgrade_fee = reader.collect_hotel_info(instance_id)
    # individual_demand_prob, individual_room_price = \
    #     reader.collect_individual_info()
    stay_sum = order_stay.sum(axis=1) * (1 - acceptance)
    lack_value = (stay_sum.reshape((-1, 1)) * order_room_quantity).sum() / (order_stay.shape[1] * capacity.sum())

    return lack_value
