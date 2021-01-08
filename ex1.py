import json
import time
from collections import defaultdict

import pandas as pd
from sklearn import cluster, manifold, mixture

from tools import fix_all_non_numeric, same_distribution, get_data, plot_scores_from_dict, run_data_analysis, \
    CLUSTERING_TO_PARAMS_MAP, plot_new_clusters, INTERNAL_CLUSTERING_LOSS

FILE_PATH = r".\data\d1_new\online_shoppers_intention.csv"
DELIMITER = ','

orig = pd.read_csv(FILE_PATH, delimiter=DELIMITER)
print(orig.shape)
print(list(orig.columns))

data_fixed = orig.copy()
fix_all_non_numeric(data_fixed)

class_key = 'VisitorType'

to_keep = {
    'Administrative',
    'Administrative_Duration',
    'Informational',
    'Informational_Duration',
    'ProductRelated',
    'ProductRelated_Duration',
    'BounceRates',
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

data_fixed = same_distribution(data_fixed, class_key)

number_classes = data_fixed[class_key].nunique()

sampled_data = data_fixed.sample(n=3000)

data, cluster_gt = get_data(data_fixed, to_keep, class_key)

CLUSTERING_TO_STEPS_MAP = {
    cluster.KMeans: range(2, 10),
    cluster.DBSCAN: [float(i)/20 for i in range(1, 20)],
    mixture.GaussianMixture: range(2, 10),
    cluster.SpectralClustering: range(2, 10),
    cluster.AgglomerativeClustering: range(2, 10)
}


CLUSTERING_ALGOS = [
    cluster.KMeans, cluster.DBSCAN, mixture.GaussianMixture, cluster.SpectralClustering, cluster.AgglomerativeClustering
]


def main():
    all_clustering_algorithms = []
    CLUSTERING_ALGOS = [cluster.SpectralClustering]
    for clusters_algo in CLUSTERING_ALGOS:
        losses_by_type_int = defaultdict(list)
        losses_by_type_ext = defaultdict(list)
        ks = []
        # k_values_to_run = CLUSTERING_TO_STEPS_MAP[clusters_algo]
        k_values_to_run = [5]

        reduction_algorithms = [manifold.TSNE]
        # k_values_to_run = [40]
        # reduction_algorithms = REDUCTION_ALGOS
        EXTERNAL_CLUSTERING_LOSS = []

        for k in k_values_to_run:
            try:
                clusters = run_data_analysis(data, k, clusters_algo)

                for reduction_algo in reduction_algorithms:
                    data_2d = reduction_algo(n_components=2).fit_transform(data)
                    plot_new_clusters(cluster_gt, clusters, data_2d)

                for loss_func in INTERNAL_CLUSTERING_LOSS:
                    loss = loss_func(data, clusters)
                    losses_by_type_int[loss_func.__name__].append(loss)

                for loss_func in EXTERNAL_CLUSTERING_LOSS:
                    loss = loss_func(cluster_gt, clusters)
                    losses_by_type_ext[loss_func.__name__].append(loss)
                ks.append(k)
            except:
                print('issue with: {}'.format(k))
        params_label = CLUSTERING_TO_PARAMS_MAP[clusters_algo]
        res_dict = (clusters_algo.__name__, {'ks': (params_label, ks), 'ext': losses_by_type_ext, 'int': losses_by_type_int})
        all_clustering_algorithms.append(res_dict)

    time_str = time.strftime('%H-%M-%S')
    file_name = '{}.json'.format(time_str)
    with open(file_name, 'w') as f:
        json.dump(all_clustering_algorithms, f, indent=4)
    print(file_name)

    plot_scores_from_dict(all_clustering_algorithms)


EX1_INT_FILE = r"C:\Workspace\Unsupervised\17-59-15.json"
EX1_EXT_FILE = r"C:\Workspace\Unsupervised\11-34-52.json"


if __name__ == '__main__':
    # run_plot_by_loss(EX1_INT_FILE)
    # run_plot_by_algo(EX1_INT_FILE)
    main()
