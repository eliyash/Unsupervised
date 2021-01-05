import pandas as pd

from tools import fix_all_non_numeric, same_distribution, run_data_analysis

FILE_PATH = r".\data\d1_new\online_shoppers_intention.csv"
DELIMITER = ','

orig = pd.read_csv(FILE_PATH, delimiter=DELIMITER)
print(orig.shape)
print(list(orig.columns))

data_fixed = orig.copy()
fix_all_non_numeric(data_fixed)

class_key = 'Revenue'
data_fixed = same_distribution(data_fixed, class_key)

to_keep = {
    # 'Administrative',
    'Administrative_Duration',
    # 'Informational',
    'Informational_Duration',
    'ProductRelated',
    'ProductRelated_Duration',
    # 'BounceRates',
    'ExitRates',
    'PageValues',
    'SpecialDay',
    'Month',
    'OperatingSystems',
    'Browser',
    'Region',
    'TrafficType',
    # 'VisitorType',
    # 'Weekend',
    # 'Revenue'
}

NUMBER_CLASSES = 10
sampled_data = data_fixed.sample(n=3000)


run_data_analysis(sampled_data, to_keep, class_key, NUMBER_CLASSES)
