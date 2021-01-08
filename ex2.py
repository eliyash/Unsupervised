import json
import time
from collections import defaultdict

import pandas as pd
from sklearn import cluster, mixture, manifold, metrics, decomposition

from tools import fix_all_non_numeric, same_distribution, run_data_analysis, fix_field, replace_unknown_with_average, \
    remove_unknown_with_ratio, get_data, plot_new_clusters, INTERNAL_CLUSTERING_LOSS, CLUSTERING_TO_PARAMS_MAP, \
    plot_scores_from_dict, CLUSTERING_TO_STEPS_MAP, run_plot_by_loss, run_plot_by_algo, run_full_flow

FILE_PATH = r".\data\d2\diabetic_data.csv"
DELIMITER = ','

to_keep = {
    'encounter_id',
    'patient_nbr',
    # 'gender',
    'age',
    'admission_type_id',
    'discharge_disposition_id',
    'admission_source_id',
    'time_in_hospital',
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'number_outpatient',
    'number_emergency',
    'number_inpatient',
    'diag_1',
    'number_diagnoses',
    'max_glu_serum',
    'A1Cresult',
    'metformin',
    'repaglinide',
    'nateglinide',
    'chlorpropamide',
    'glimepiride',
    'acetohexamide',
    'glipizide',
    'glyburide',
    'tolbutamide',
    'pioglitazone',
    'rosiglitazone',
    'acarbose',
    'miglitol',
    'troglitazone',
    'tolazamide',
    'examide',
    'citoglipton',
    'insulin',
    'glyburide-metformin',
    'glipizide-metformin',
    'glimepiride-pioglitazone',
    'metformin-rosiglitazone',
    'metformin-pioglitazone',
    'change',
    'diabetesMed',
    'readmitted'
}

CLUSTERING_TO_STEPS_MAP_SMALL = {
    cluster.KMeans: range(2, 15),
    cluster.DBSCAN: [float(i)/20 for i in range(1, 10)],
    mixture.GaussianMixture: range(2,15),
    cluster.SpectralClustering: range(2, 5),
    cluster.AgglomerativeClustering: range(2, 15)
}


CLUSTERING_ALGOS = [
    cluster.KMeans, cluster.DBSCAN, mixture.GaussianMixture, cluster.SpectralClustering, cluster.AgglomerativeClustering
]


# EX1_INT_FILE = r"C:\Workspace\Unsupervised\17-59-15.json"
# EX1_EXT_FILE = r"C:\Workspace\Unsupervised\11-34-52.json"

def main():
    orig = pd.read_csv(FILE_PATH, delimiter=DELIMITER)
    print(orig.shape)

    data_fixed = orig.copy()
    ratio_unknown = 0.5
    ages = range(0, 130, 10)
    fix_field(data_fixed, 'age', ['[{}-{})'.format(age, age + 10) for age in ages], [i for i in range(len(ages))])
    remove_unknown_with_ratio(data_fixed, ratio_unknown)
    fix_all_non_numeric(data_fixed)
    replace_unknown_with_average(data_fixed)

    class_key = 'gender'
    data_fixed = same_distribution(data_fixed, class_key)
    data_fixed = data_fixed.sample(n=2000)

    data, cluster_gt = get_data(data_fixed, to_keep, class_key)
    internal_clustering_loss = [metrics.silhouette_score]
    external_clustering_loss = [metrics.mutual_info_score]
    reduction_algorithms = [manifold.TSNE]
    pre_clustering_reduction_algorithm = decomposition.PCA(15)
    clusters_algos = CLUSTERING_ALGOS

    internal_clustering_loss = []
    reduction_algorithms = []

    if pre_clustering_reduction_algorithm:
        data = pre_clustering_reduction_algorithm.fit_transform(data)

    clusters_params_zip = [(algo, CLUSTERING_TO_STEPS_MAP_SMALL[algo]) for algo in clusters_algos]
    file_name, all_clustering_algorithms = run_full_flow(
        data, cluster_gt, clusters_params_zip, internal_clustering_loss, external_clustering_loss, reduction_algorithms
    )
    print(file_name)

    run_plot_by_loss(file_name)
    # plot_scores_from_dict(all_clustering_algorithms)
    # run_plot_by_algo(file_name)


if __name__ == '__main__':
    main()
