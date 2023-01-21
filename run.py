ads_within_budget = [{'fee_per_job': 86.022, 'capacity': 309, 'processor_id': 'P_low_0'}, {'fee_per_job': 90.54180000000001, 'capacity': 873, 'processor_id': 'P_low_1'}, {'fee_per_job': 93.1662, 'capacity': 412, 'processor_id': 'P_low_2'}, {'fee_per_job': 80.19000000000001, 'capacity': 496, 'processor_id': 'P_low_7'}, {'fee_per_job': 95.65938000000003, 'capacity': 914, 'processor_id': 'P_low_9'}, {'fee_per_job': 93.29742000000002, 'capacity': 500, 'processor_id': 'P_low_11'}, {'fee_per_job': 78.732, 'capacity': 937, 'processor_id': 'P_low_12'}, {'fee_per_job': 86.022, 'capacity': 721, 'processor_id': 'P_low_16'}, {'fee_per_job': 83.10600000000001, 'capacity': 353, 'processor_id': 'P_low_17'}, {'fee_per_job': 82.62, 'capacity': 647, 'processor_id': 'P_low_19'}]



def has_min_reputation(): 
    return 2> 1


matches = [ad for ad in ads_within_budget if has_min_reputation()]


print(matches)