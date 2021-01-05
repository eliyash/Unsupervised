import pandas as pd

from tools import fix_all_non_numeric, same_distribution, run_data_analysis, fix_field, replace_unknown_with_average, \
    remove_unknown_with_ratio

FILE_PATH = r".\data\d2\diabetic_data.csv"
DELIMITER = ','

orig = pd.read_csv(FILE_PATH, delimiter=DELIMITER)
print(orig.shape)
print(list(orig.columns))

data_fixed = orig.copy()
ratio_unknown = 0.5
ages = range(0, 130, 10)
fix_field(data_fixed, 'age', ['[{}-{})'.format(age, age+10) for age in ages], [i for i in range(len(ages))])
remove_unknown_with_ratio(data_fixed, ratio_unknown)
fix_all_non_numeric(data_fixed)
replace_unknown_with_average(data_fixed)

class_key = 'gender'
data_fixed = same_distribution(data_fixed, class_key)

to_keep = {
    'encounter_id',
    'patient_nbr',
    # 'gender',
    'age',
    # 'admission_type_id',
    'discharge_disposition_id',
    # 'admission_source_id',
    'time_in_hospital',
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    # 'number_outpatient',
    # 'number_emergency',
    # 'number_inpatient',
    'diag_1',
    # 'number_diagnoses',
    # 'max_glu_serum',
    # 'A1Cresult',
    # 'metformin',
    # 'repaglinide',
    # 'nateglinide',
    # 'chlorpropamide',
    # 'glimepiride',
    # 'acetohexamide',
    # 'glipizide',
    # 'glyburide',
    # 'tolbutamide',
    # 'pioglitazone',
    # 'rosiglitazone',
    # 'acarbose',
    # 'miglitol',
    # 'troglitazone',
    # 'tolazamide',
    # 'examide',
    # 'citoglipton',
    # 'insulin',
    # 'glyburide-metformin',
    # 'glipizide-metformin',
    # 'glimepiride-pioglitazone',
    # 'metformin-rosiglitazone',
    # 'metformin-pioglitazone',
    # 'change',
    # 'diabetesMed',
    # 'readmitted'
}

NUMBER_CLASSES = 10
sampled_data = data_fixed.sample(n=3000)


run_data_analysis(sampled_data, to_keep, class_key, NUMBER_CLASSES)
