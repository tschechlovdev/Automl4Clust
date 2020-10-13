import logging
import math
import time

from Algorithm import ClusteringAlgorithms
from MetaLearning import MetaFeatureExtractor, KNNSearch
from Metrics.MetricHandler import MetricCollection
from Optimizer.Optimizer import RandomOptimizer, SMACOptimizer, HyperBandOptimizer, BOHBOptimizer
from Utils import Constants
from Utils.Constants import REAL_WORLD, SYNTHETIC
from Utils.FileUtil import FileImporter, FileExporter
import numpy as np
import pandas as pd


def get_top_n_metric_results(metric_result_df, n_warmstarts, config_space):
    k_range = config_space[0]
    # Check if configs are in config space
    if not "algorithm" in metric_result_df.columns:
        metric_result_df["algorithm"] = ClusteringAlgorithms.KMEANS_ALGORITHM
    # filter configs to only include k values in the k range
    metric_result_df = metric_result_df[(metric_result_df["k"] >= k_range[0]) & (metric_result_df["k"] <= k_range[1])]

    metric_result_df = metric_result_df.drop_duplicates(subset=["k", "algorithm"], keep='last')
    if len(metric_result_df.index) >= n_warmstarts:
        metric_result_df = metric_result_df.sort_values('score', ascending=True).head(n_warmstarts)

    warmstart_configs = []
    for index, row in metric_result_df.iterrows():
        warmstart_configs.append((row['k'], row['algorithm']))

    return warmstart_configs


def get_warmstarts(n_warmstarts, online_data, online_data_labels, optimizers, metrics, experiment, config_space,
                   n_datasets=1, dataset_type=SYNTHETIC):
    time_mf_extraction = time.time()
    # extract metafeatures and find nearest neighbor
    # now extract the metafeatures from the online data sets
    if isinstance(online_data, pd.DataFrame):
        online_data = online_data.to_numpy()
    online_metafeatures_dic = MetaFeatureExtractor.extract_meta_features_as_dic(data=online_data,
                                                                                data_labels=online_data_labels,
                                                                                phase=Constants.ONLINE_PHASE,
                                                                                dataset_type=dataset_type)
    logging.info("finished meta_feature extraction")
    time_mf_extraction = time.time() - time_mf_extraction

    nn_start = time.time()
    logging.info("finding nn for dataset_Type {}....".format(dataset_type))
    mfs_array = np.fromiter(online_metafeatures_dic.values(), dtype=float).reshape(1, -1)
    nn_datasets = KNNSearch.find_nearest_neighbors(mfs_array, k=n_datasets, dataset_type=dataset_type)
    logging.info("found nn: {}".format(nn_datasets))
    nn_execution_time = time.time() - nn_start
    result_dic = {"mf_time": time_mf_extraction, "nn_time": nn_execution_time}

    for optimizer in optimizers:
        # list of all metric results (the best! results) for this specific optimizer
        result_dic_per_optimizer = {}
        for metric in metrics:
            metric_name = metric.name
            result_dic_per_metric = {}
            search_config_start = time.time()
            metric_result_dfs = []
            for nn_dataset in nn_datasets:
                logging.info(nn_datasets)
                metric_result_dfs.append(
                    FileImporter.import_result_for_phase(filename=nn_dataset, optimizer_name=optimizer.get_name(),
                                                         metric_name=metric_name, experiment=experiment, repetition=0,
                                                         dataset_type=dataset_type))
            # get all metric executions for this internal metric
            metric_result_df_all_datasets = pd.concat(metric_result_dfs)
            warm_start_configs = get_top_n_metric_results(metric_result_df_all_datasets, n_warmstarts, config_space)
            config_time = time.time() - search_config_start
            result_dic_per_metric["warmstarts"] = warm_start_configs
            result_dic_per_metric["config time"] = config_time
            logging.info("Best configs for optimizer {} and metric {}: {}".format(optimizer.get_name(), metric_name,
                                                                                  warm_start_configs))
            result_dic_per_optimizer[metric_name] = result_dic_per_metric
        result_dic[optimizer.get_name()] = result_dic_per_optimizer

    return result_dic
