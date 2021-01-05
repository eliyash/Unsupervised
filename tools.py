import random
from statistics import median
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics, neighbors, preprocessing, cluster, decomposition, manifold, mixture

CLUSTERS_ALGOS = [
    cluster.KMeans,
    mixture.GaussianMixture, cluster.SpectralClustering, cluster.AgglomerativeClustering, cluster.OPTICS, cluster.DBSCAN
]

# a = mixture.GaussianMixture().fit_predict()
# a = cluster.SpectralClustering().fit_predict()
# a = cluster.AgglomerativeClustering().fit_predict()
# a = cluster.OPTICS().fit_predict()
# a = cluster.DBSCAN().fit_predict()
# a = cluster.KMeans().fit_predict()


REDUCTION_ALGOS = [
    decomposition.PCA, decomposition.TruncatedSVD, decomposition.FastICA,
    decomposition.KernelPCA, manifold.Isomap, manifold.TSNE, manifold.MDS
]

SIZE_OF_SCATTER = 2


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


def calculate_clusters_and_plot(
        orig_data, to_keep, real_clusters_key, clustering_algo=None, reduction_algo=None, num_clusters=2
):
    data_dropped = drop_keys(orig_data, to_keep)

    normalised_data, no_outlier_map = remove_outliers_and_normalize(data_dropped)

    res_data = normalised_data
    if clustering_algo:
        model = clustering_algo(num_clusters)
        #     model.fit(normalised_data)
        #     full_reduced_data = model.transform(normalised_data)
        #     print(model.explained_variance_ratio_)
        #     reduced_data = full_reduced_data[:, 0:2]
        # clusters_wights = model.transform(res_data)
        clusters = model.fit_predict(res_data)
        # clusters_wights = model.fit_predict(res_data)
        # clusters = np.argmax(clusters_wights, axis=1)
    else:
        clusters = []

    if reduction_algo:
        model = reduction_algo(n_components=2)
        res_data = model.fit_transform(res_data)

    x_range, y_range = map(tuple, zip(res_data.min(axis=0), res_data.max(axis=0)))

    cluster_gt = orig_data[real_clusters_key][no_outlier_map > 0]

    plt.figure(figsize=(15, 10))
    lines = 4
    # plt.subplots(2, 1, constrained_layout=True)
    # plt.title('clustering alg: {}, reduction alg: {}, clusters: {}'.format(clustering_algo.__name__, reduction_algo.__name__, num_clusters))
    scatter(res_data, x_range, y_range, (lines, 1), 1, {'c': cluster_gt})
    scatter(res_data, x_range, y_range, (lines, 1), 2, {'c': clusters})
    scatter_by_key(res_data, np.array(cluster_gt), x_range, y_range, lines, 3)
    scatter_by_key(res_data, clusters, x_range, y_range, lines, 4)
    plt.show()


def plot_pca_without_values_with_any_key(orig_data, clustering_algo=None, reduction_algo=None, num_clusters=2):
    all_keys = set(orig_data.columns)
    for key in all_keys:
        if orig_data[key].nunique() < 6:
            key_to_keep = all_keys - {key}
            key_data = same_distribution(orig_data, key)
            calculate_clusters_and_plot(key_data, key_to_keep, key, clustering_algo, reduction_algo, num_clusters)


def run_data_analysis(sampled_data, to_keep, class_key, number_classes):
    for reduction_algo in REDUCTION_ALGOS:
        for clustering_algo in CLUSTERS_ALGOS:
            calculate_clusters_and_plot(
                sampled_data, to_keep, class_key, clustering_algo, reduction_algo, number_classes
            )


def calc_clustering_loss(orig_data, algo):
    metrics.log_loss
    metrics.mutual_info_score
    metrics.auc
    metrics.silhouette_score
    # Dunn index

    # Non – parametric tests
    # T - test – Mann whitney(U test)
    # ANOVA – Kruskal Wallis

    #P value must be reported.
