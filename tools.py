import json
import random
import time
from collections import defaultdict
from statistics import median
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from sklearn import metrics, neighbors, preprocessing

from scipy import stats
from sklearn import cluster, decomposition, manifold, mixture

# CLUSTERING_LOSS = [
#     metrics.mutual_info_score,
#     metrics.log_loss,
#     metrics.normalized_mutual_info_score,
#     metrics.v_measure_score
# ]
#
# SELF_CLUSTERING_LOSS = [
#     metrics.silhouette_score, stats.f_oneway, stats.kruskal, stats.spearmanr
# ]
#
# CLUSTERS_ALGOS = [
#     cluster.KMeans, mixture.GaussianMixture, cluster.SpectralClustering, cluster.AgglomerativeClustering, cluster.DBSCAN
# ]
#
# REDUCTION_ALGOS = [
#     manifold.TSNE,
#     decomposition.PCA,
#     decomposition.TruncatedSVD, decomposition.FastICA,
#     decomposition.KernelPCA, manifold.Isomap, manifold.MDS
# ]

SIZE_OF_SCATTER = 2


class FixLossKeepNameBase:
    def __init__(self, func):
        self._func = func
        self.__name__ = func.__name__

    @staticmethod
    def pre_process(*args, **kwargs):
        return args, kwargs

    @staticmethod
    def post_process(res):
        return res

    def __call__(self, *args, **kwargs):
        args, kwargs = self.pre_process(*args, **kwargs)
        res = self._func(*args, **kwargs)
        res = self.post_process(res)
        return res


class kruskal_fixed(FixLossKeepNameBase):
    def __init__(self):
        super().__init__(stats.kruskal)

    @staticmethod
    def pre_process(data, clusters):
        x = [data[clusters == cluster_val] for cluster_val in set(clusters)]
        return x, {}

    @staticmethod
    def post_process(res):
        return res.pvalue


class f_oneway_fixed(FixLossKeepNameBase):
    def __init__(self):
        super().__init__(stats.f_oneway)

    @staticmethod
    def pre_process(data, clusters):
        x = [data[clusters == cluster_val] for cluster_val in set(clusters)]
        return x, {}

    @staticmethod
    def post_process(res):
        return max(res.pvalue)


class fuzzy_c_means:
    def __init__(self, n_centers):
        self._n_centers = n_centers

    def fit_predict(self, data):
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data, self._n_centers, 2, error=0.005, maxiter=1000, init=None)
        return np.argmax(u, axis=0)



def check_data_info(data, types=None):
    if not types:
        types = [int, float]
    is_numeric = {}
    diff_values = {}
    count_unknown = {}
    for column in data.columns:
        is_numeric[column] = not any([not type(val) in types for val in data[column]])
        diff_values[column] = len({val for val in data[column]})
        count_unknown[column] = len([val for val in data[column] if val == '?'])
    return is_numeric, diff_values, count_unknown


def rand_colors(count):
    random.seed(0)
    colors = [tuple(map(lambda _: random.random(), range(3))) for _ in range(count)]
    return colors


def fix_field(data, field, orig, dest):
    data[field].replace(orig, dest, inplace=True)


def fix_field_auto(data, field):
    names_map = get_new_map(data, field)
    old_names = names_map.keys()
    new_names = [names_map[old_name] for old_name in old_names]
    fix_field(data, field, old_names, new_names)
    return names_map


def remove_unknown_with_ratio(data, ratio_unknown):
    max_count = len(data.columns) * ratio_unknown
    _, _, count_unknown = check_data_info(data)
    keys_to_remove = {key for key, count in count_unknown.items() if count > max_count}
    data.drop(columns=keys_to_remove, inplace=True)


def get_min_max(data):
    min_max = {}
    for key in data.columns:
        data_no_question_mark = data[key].replace('?', 0)
        min_max[key] = (data_no_question_mark.min(), data_no_question_mark.max())
    return min_max


def replace_unknown_with_average(data):
    min_max_dict = get_min_max(data)
    for key, min_max in min_max_dict.items():
        data[key].replace('?', sum(min_max) / 2, inplace=True)


def fix_all_non_numeric(data):
    is_numeric_dict, _, _ = check_data_info(data)
    [fix_field_auto(data, key) for key, is_numeric in is_numeric_dict.items() if not is_numeric]


def remove_all_non_numeric(data):
    is_numeric_dict, _, _ = check_data_info(data)
    keys_to_remove = {key for key, is_numeric in is_numeric_dict.items() if not is_numeric}
    data.drop(columns=keys_to_remove, inplace=True)


def get_new_map(data, field):
    return {name: '?' if name == '?' else ind for ind, name in enumerate(set(data[field]))}


def same_distribution(data, field):
    values_map = dict(data[field].value_counts())
    median_val = int(median(values_map.values()))
    new_dataframe = pd.DataFrame()
    for key in values_map.keys():
        new_val = min(values_map[key], median_val)
        new_key_dataframe = data.loc[data[field] == key].sample(new_val)
        new_dataframe = pd.concat([new_dataframe, new_key_dataframe])
    return new_dataframe


def drop_keys(orig_data, to_keep):
    remove = set(orig_data.columns) - to_keep
    return orig_data.drop(columns=remove)


def scatter(data, x_range, y_range, shape, loc, slot_kwargs):
    x = data[:, 0]
    y = data[:, 1]
    plt.subplot(*shape, loc)
    plt.scatter(x, y, s=SIZE_OF_SCATTER, **slot_kwargs)
    plt.xlim(x_range)
    plt.ylim(y_range)


def scatter_by_key(reduced_data, cluster_gt, x_range, y_range, lines, line):
    unique_keys = set(cluster_gt)
    size = len(unique_keys)
    colors = rand_colors(size)
    loc = 1
    for i, key in enumerate(unique_keys):
        reduced_data_of_key = reduced_data[cluster_gt == key]
        scatter(reduced_data_of_key, x_range, y_range, (lines, size), size*(line-1) + loc, {'color': colors[i]})
        loc += 1


def check_quality(data, to_compare_key):
    data = data.copy()
    scores = {}
    fix_field_auto(data, to_compare_key)

    for key in data.columns:
        scores[key] = metrics.mutual_info_score(data[key].to_numpy(), data[to_compare_key].to_numpy())
    return scores


def remove_outliers_and_normalize(data_dirty, n_neighbors=20):
    clf = neighbors.LocalOutlierFactor(n_neighbors=n_neighbors)
    norm = preprocessing.Normalizer()
    data_map = clf.fit_predict(data_dirty)
    data_clean = data_dirty[data_map > 0]
    data_normalized = norm.fit_transform(data_clean)
    return data_normalized, data_map


def plot_new_clusters(cluster_gt, clusters, data):
    x_range, y_range = map(tuple, zip(data.min(axis=0), data.max(axis=0)))
    plt.figure(figsize=(15, 10))
    lines = 4
    scatter(data, x_range, y_range, (lines, 1), 1, {'c': cluster_gt})
    scatter(data, x_range, y_range, (lines, 1), 2, {'c': clusters})
    scatter_by_key(data, np.array(cluster_gt), x_range, y_range, lines, 3)
    scatter_by_key(data, clusters, x_range, y_range, lines, 4)
    plt.show()


CLUSTERING_TO_PARAMS_MAP = {
    cluster.KMeans: 'clusters',
    cluster.DBSCAN: 'max neighbor distance',
    mixture.GaussianMixture: 'clusters',
    cluster.SpectralClustering: 'clusters',
    cluster.AgglomerativeClustering: 'clusters',
    fuzzy_c_means: 'clusters',
}


def plot_scores(title, x_label, res, is_bar=False):
    if is_bar:
        plt.bar([name for name, losses in res], [min(losses) for name, losses in res])
    else:
        for name, ks, losses in res:
            plt.plot(ks, losses, label=name)
        plt.xlabel(x_label)
        plt.ylabel('loss')
        plt.legend()
    plt.title(title)
    plt.show()


def get_data(orig_data, to_keep, real_clusters_key):
    data_dropped = drop_keys(orig_data, to_keep)
    res_data, no_outlier_map = remove_outliers_and_normalize(data_dropped)
    cluster_gt = np.array(orig_data[real_clusters_key][no_outlier_map > 0])
    return res_data, cluster_gt


def calculate_clusters_and_plot(orig_data, num_clusters, clustering_algo):
    data = orig_data.copy()
    clusters = clustering_algo(num_clusters).fit_predict(data)
    return clusters


def run_data_analysis(data, num_clusters, clustering_algo):
    clusters = calculate_clusters_and_plot(data, num_clusters, clustering_algo)
    print('clustering alg: {}, clusters: {}'.format(clustering_algo.__name__, num_clusters))

    return clusters


def load_chosen(path):
    with open(path) as f:
        return json.load(f)


def plot_scores_from_dict(data_list):
    for cluster_algo, res in data_list:
        title = 'clustering algorithm: {}'.format(cluster_algo)
        label, ks = res['ks']
        ext_dict = res['ext']
        int_dict = res['int']
        if ext_dict:
            plot_scores(title, label, [(name, ks, extern) for name, extern in ext_dict.items()])
        if int_dict:
            plot_scores(title, label, [(name, ks, inter) for name, inter in int_dict.items()])


def run_plot_by_algo(path):
    all_clustering_algorithms = load_chosen(path)
    plot_scores_from_dict(all_clustering_algorithms)


def run_plot_by_loss(path):
    all_clustering_algorithms = load_chosen(path)
    data_by_losses = defaultdict(dict)
    x_label_by_clustering = defaultdict()
    for cluster_algo, res in all_clustering_algorithms:
        x_label_by_clustering[cluster_algo] = res['ks']
        for name, internal in res['int'].items():
            data_by_losses[name][cluster_algo] = internal
        for name, external in res['ext'].items():
            data_by_losses[name][cluster_algo] = external

    for loss_name, loss_data in data_by_losses.items():
        title = 'loss algorithm: {}'.format(loss_name)
        plot_scores(title, '', [(clustering_name, values) for clustering_name, values in loss_data.items()], is_bar=True)

EXTERNAL_CLUSTERING_LOSS = [
    metrics.mutual_info_score,
    metrics.v_measure_score
]

INTERNAL_CLUSTERING_LOSS = [
    metrics.silhouette_score,
    # f_oneway_fixed(),
]


REDUCTION_ALGOS = [
    manifold.TSNE,
    decomposition.PCA,
    # decomposition.TruncatedSVD, decomposition.FastICA,
    # decomposition.KernelPCA, manifold.Isomap, manifold.MDS
]

CLUSTERING_TO_STEPS_MAP = {
    cluster.KMeans: range(2, 70),
    cluster.DBSCAN: [float(i)/200 for i in range(1, 200)],
    mixture.GaussianMixture: range(2, 50),
    cluster.SpectralClustering: range(2, 45),
    cluster.AgglomerativeClustering: range(2, 80),
    # fuzzy_c_means: range(2, 70),
}

def run_full_flow(data, cluster_gt, clusters_params_zip, internal_clustering_loss, external_clustering_loss, reduction_algorithms):
    all_clustering_algorithms = []
    for clusters_algo, k_values_to_run in clusters_params_zip:
        losses_by_type_int = defaultdict(list)
        losses_by_type_ext = defaultdict(list)
        ks = []
        for k in k_values_to_run:
            try:
                clusters = run_data_analysis(data, k, clusters_algo)

                for reduction_algo in reduction_algorithms:
                    data_2d = reduction_algo(n_components=2).fit_transform(data)
                    plot_new_clusters(cluster_gt, clusters, data_2d)

                for loss_func in internal_clustering_loss:
                    loss = loss_func(data, clusters)
                    losses_by_type_int[loss_func.__name__].append(loss)

                for loss_func in external_clustering_loss:
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
    return file_name, all_clustering_algorithms