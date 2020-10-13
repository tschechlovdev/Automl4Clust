"""
This class contains code that is related to every kind of file interaction.
This includes a FileImporter, which imports the datasets that are used for the experiments as well as a FileExporter.
This ensures a holistic way of interacting with the files.
It also contains some utility functions.
"""

import logging
import os

import joblib
import pandas as pd
import numpy as np
import re
# from sklearn.externals import joblib

#from Algorithm import ClusteringAlgorithms
from Algorithm import ClusteringAlgorithms
from Metrics.MetricHandler import MetricCollection
from Utils import Constants
import Definitions
from Utils.Constants import *

DEFAULT_DELIMITER = ";"
DEFAULT_DECIMAL = ","
PROJECT_PATH = Definitions.ROOT_DIR
PROJECT_RESULTS_PATH = PROJECT_PATH + "/results/result"
RESULTS_PATH_REAL_WORLD = PROJECT_RESULTS_PATH + "/" + REAL_WORLD
RESULTS_ONLINE_TEMPLATE = PROJECT_RESULTS_PATH + "/{experiment}/online/{warmOrColdstart}"

RESULTS_OFFLINE_TEMPLATE = PROJECT_RESULTS_PATH + "/{experiment}/offline/{n_iterations}"

RESULTS_ONLINE_TEMPLATE_REAL_WORLD = PROJECT_RESULTS_PATH + "/" + REAL_WORLD + "/{experiment}/online/{warmOrColdstart}"

DATASET_PATH = PROJECT_PATH
RESULTS_INTERNAL_METRICS = PROJECT_RESULTS_PATH + "/metrics/online/{algorithm}/{repetition}/"
RESULTS_METRICS_GENERAL = PROJECT_RESULTS_PATH + "/metrics/online/{metric}/{repetition}/{filename}"
META_FEATURE_PATH = PROJECT_RESULTS_PATH + "/metafeatures/"
META_FEATURE_PATH_FERRARI = PROJECT_RESULTS_PATH + "/metafeatures/ferrari/"
optimizers_merged_filename = "optimizer_results_merged.csv"
optimizer_export_path = RESULTS_ONLINE_TEMPLATE
optimizers_merged_filename_external = "external_optimizer_results_merged.csv"

META_FEATURE_PATH_REAL_WORLD = PROJECT_RESULTS_PATH + "/" + REAL_WORLD + "/metafeatures"

DEFAULT_METRICS_PATH = PROJECT_RESULTS_PATH +"/"+  REAL_WORLD + "/metrics/online"
METRIC_EXPORT_PATH_TEMPLATE = DEFAULT_METRICS_PATH + "/{algorithm}/{repetition}/"

class FileImporter:
    REAL_WORLD_DATA_PATH = "/volume/data/"
    PATH_TO_OFFLINE_DATA = DATASET_PATH + "/Offline_datasets"
    PATH_TO_ONLINE_DATA = DATASET_PATH + "/Online_datasets"

    @staticmethod
    def get_file_list(path):
        return os.listdir(path)

    @staticmethod
    def get_offline_file_list(sorting=False, reverse=False, path_to_files=PATH_TO_OFFLINE_DATA):
        """
        Return names of all files in the file path.
        If file path is not specified then the default directory to the synthetic dataset will be used.
        :param path_to_files:
        :param sorting:
        :param reverse:
        :return: List of strings containing file names (including path to the files)
        """
        result = os.listdir(path_to_files)
        if sorting:
            return sorted(result, reverse=reverse)
        else:
            return result

    @staticmethod
    def import_data(filepath):
        """
        Imports the data that is specified by filepath. It returns a tuple that has first the dataset as np.array
        without labels and as second entry the labels of the dataset.
        :param filepath: path to csv file of the dataset.
        :return: tuple containing dataset without labels and the true_labels
        """
        logging.info("Importing {}".format(filepath))
        raw_data = pd.read_csv(filepath, header=None)
        logging.info("Imported {}, shape is {}".format(filepath, raw_data.shape))
        n_features = raw_data.shape[1]
        true_labels = np.array(raw_data[n_features - 1])
        data_without_labels = raw_data[raw_data.columns[:-1]]
        logging.info("data (without labels) has shape {}".format(data_without_labels.shape))
        return data_without_labels, true_labels

    @staticmethod
    def import_offline_data(filename, file_path=None):
        path = file_path if file_path else FileImporter.PATH_TO_OFFLINE_DATA
        return FileImporter.import_data(os.path.join(path, filename))

    @staticmethod
    def import_data_as_df(filepath, filename=None, decimal=","):
        if filename is None:
            return pd.read_csv(filepath, delimiter=DEFAULT_DELIMITER, decimal=decimal)
        else:
            return pd.read_csv(os.path.join(filepath, filename), delimiter=DEFAULT_DELIMITER, decimal=decimal)

    @staticmethod
    def import_raw_data(filepath,header=None, sep=None):
        """
        Just imports the data from the specified pat to the csv file.
        So the data will not be separated into data and labels.
        :param filepath:
        :return: dataset in its raw format as in the csv file.
        """
        return pd.read_csv(filepath, header=header, sep=sep)

    @staticmethod
    def load_model(path, file):
        return joblib.load(os.path.join(path, file))

    @staticmethod
    def import_online_datasets(file_path=None):
        """
        Imports the online datasets. Each dataset is imported as Dataframe.

        :type file_path: str
        :param file_path: datasets will be imported from this path if given. Otherwise the default path will be used.
        :return: list of Dataframe object. One Dataframe for each dataset.
        """
        path_to_online_data = file_path if file_path else FileImporter.PATH_TO_ONLINE_DATA
        return FileImporter.import_datasets_df(path_to_online_data)

    @staticmethod
    def import_online_dataset(filename, filepath=None):
        file_path = FileImporter.PATH_TO_ONLINE_DATA if not filepath else filepath
        return FileImporter.import_offline_data(filename, file_path)

    @staticmethod
    def import_datasets_df(path):
        """
        Imports the datasets in the specified path as dataframe objects.
        The result will be a list of Dataframe objects, one for each dataset in the directory.
        The directory has to contain the datasets as csv files.
        :param path: path to the datasets
        :return: list of Dataframe objects
        """
        datasets = []
        filenames = os.listdir(path)
        for file in filenames:
            df = FileImporter.import_data_as_df(path, file)
            datasets.append(df)
        return datasets

    @staticmethod
    def list_online_data(file_path=None, dataset_type=SYNTHETIC):
        if dataset_type == REAL_WORLD:
            return [  x for x in os.listdir(FileImporter.REAL_WORLD_DATA_PATH) if not "madelon" in x]
        path = file_path if file_path else FileImporter.PATH_TO_ONLINE_DATA
        return os.listdir(path)

    @staticmethod
    def get_offline_results_path(experiment=HPO_EXPERIMENT, n_iterations="n_div_40"):
        """
        Returns the path where the optimizer results are contained in general. Does not return the path for a specific
        optimizer or metric.
        :param warmOrColdstart:
        :param experiment:
        :return:
        """
        return RESULTS_OFFLINE_TEMPLATE.format(experiment=experiment, n_iterations=n_iterations)

    @staticmethod
    def get_offline_optimizer_results_path(optimizer_name, metric_name, experiment=HPO_EXPERIMENT,
                                           n_iterations="n_div_40", repetition=0):
        """
        Get the path where the optimizer results are contained.
        :param optimizer_name:
        :param metric_name:
        :param experiment:
        :param n_iterations:
        :return:
        """
        return FileImporter.get_offline_results_path(experiment, n_iterations) + "/{opt}/{metric}/{repetition}" \
            .format(opt=optimizer_name, metric=metric_name, repetition=repetition)

    @staticmethod
    def get_online_optimizer_results_path(optimizer_name, metric_name, repetition=None,
                                          experiment=HPO_EXPERIMENT,
                                          init_strategy=None, dataset_type=REAL_WORLD, t=1, n_warmstarts=None):
        """
        Get the path where the optimizer results are contained.
        If repetition is None, then we do go to the folder one above where the aggregations are saved.
        :param init_strategy:
        :param optimizer_name:
        :param metric_name:
        :param repetition:
        :param warmOrColdstart:
        :param experiment:
        :return:
        """
        if repetition is None:
            return FileImporter.get_online_results_path(init_strategy, experiment) + "/{optimizer_name}/{metric_name}" \
                .format(optimizer_name=optimizer_name, metric_name=metric_name)
        if t == 1 or t == None:
            if dataset_type == REAL_WORLD:
                return FileImporter.get_online_results_path(init_strategy,
                                                            experiment, dataset_type=dataset_type) + "/{optimizer_name}/{metric_name}/{repetition}" \
                           .format(optimizer_name=optimizer_name, metric_name=metric_name, repetition=repetition,
                                   dataset_type=dataset_type)
            else:
                return FileImporter.get_online_results_path(init_strategy,
                                                            experiment, dataset_type=dataset_type) + "/{optimizer_name}/{metric_name}/{repetition}" \
                           .format(optimizer_name=optimizer_name, metric_name=metric_name, repetition=repetition)
        else:
            if dataset_type == REAL_WORLD:
                return FileImporter.get_online_results_path(init_strategy,
                                                            experiment, dataset_type=dataset_type) + "/{optimizer_name}/{metric_name}/{repetition}" \
                           .format(optimizer_name=optimizer_name, metric_name=metric_name, repetition=repetition,
                                   dataset_type=dataset_type)
            else:
                return FileImporter.get_online_results_path(init_strategy,
                                                            experiment, dataset_type=dataset_type) + "/{optimizer_name}/{metric_name}/{repetition}" \
                           .format(optimizer_name=optimizer_name, metric_name=metric_name, repetition=repetition)
    @staticmethod
    def get_online_results_path(warmOrColdstart=Constants.COLDSTART_STRATEGY, experiment=HPO_EXPERIMENT,
                                dataset_type=REAL_WORLD):
        """
        Returns the path where the optimizer results are contained in general. Does not return the path for a specific
        optimizer or metric.
        :param warmOrColdstart:
        :param experiment:
        :return:
        """
        if dataset_type == REAL_WORLD:
            return RESULTS_ONLINE_TEMPLATE_REAL_WORLD.format(experiment=experiment, warmOrColdstart=warmOrColdstart)
        return RESULTS_ONLINE_TEMPLATE.format(experiment=experiment, warmOrColdstart=warmOrColdstart)

    @staticmethod
    def import_offline_result(filename, optimizer_name, metric_name, experiment, repetition=0, dataset_type=SYNTHETIC):
        if dataset_type == REAL_WORLD:
            path = FileImporter.get_online_optimizer_results_path(optimizer_name, metric_name, repetition, experiment,
                                                                  init_strategy=COLDSTART_STRATEGY,
                                                                  dataset_type=dataset_type)
        else:
            path = FileImporter.get_offline_optimizer_results_path(optimizer_name=optimizer_name,
                                                                   metric_name=metric_name, experiment=experiment,
                                                                   repetition=repetition)
        return FileImporter.import_data_as_df(path, filename)

    @staticmethod
    def import_online_result(filename, optimizer_name, metric_name, experiment,
                             init_strategy=Constants.COLDSTART_STRATEGY, repetition=0):
        path = FileImporter.get_online_optimizer_results_path(optimizer_name=optimizer_name,
                                                              metric_name=metric_name, experiment=experiment,
                                                              repetition=repetition, init_strategy=init_strategy)
        return FileImporter.import_data_as_df(path, filename)

    @staticmethod
    def import_result_for_phase(filename, optimizer_name, metric_name, experiment, ex_phase=Constants.ONLINE_PHASE,
                                init_strategy=Constants.COLDSTART_STRATEGY, repetition=0, t=None, n_warmstart=None,
                                dataset_type=SYNTHETIC):
        if ex_phase == Constants.ONLINE_PHASE:
            path = FileImporter.get_online_optimizer_results_path(optimizer_name=optimizer_name,
                                                                  metric_name=metric_name, experiment=experiment,
                                                                  repetition=repetition, init_strategy=init_strategy,
                                                                  t=t, n_warmstarts=n_warmstart,
                                                                  dataset_type=dataset_type)
            print(path)
            return FileImporter.import_data_as_df(path, filename)
        else:
            path = FileImporter.get_offline_optimizer_results_path(optimizer_name=optimizer_name,
                                                                   metric_name=metric_name, experiment=experiment,
                                                                   repetition=repetition)
            return FileImporter.import_data_as_df(path, filename)

    @staticmethod
    def import_metric_result(file_name, rep, algo, dataset_type=SYNTHETIC):
        if dataset_type == SYNTHETIC:
            path = RESULTS_INTERNAL_METRICS.format(repetition=rep, algorithm=algo)
        else:
            path = METRIC_EXPORT_PATH_TEMPLATE.format(algorithm=algo, repetition=rep)
        if os.path.isdir(path):
            return FileImporter.import_data_as_df(filepath=path, filename=file_name)
        else:
            print("path {} does not exist!".format(path))
            return pd.DataFrame()

    @staticmethod
    def list_data_for_phase(experiment_phase, dataset_type=Constants.SYNTHETIC):
        if experiment_phase == Constants.ONLINE_PHASE:
            return FileImporter.list_online_data(dataset_type=dataset_type)
        elif experiment_phase == Constants.OFFLINE_PHASE:
            return FileImporter.get_offline_file_list()
        else:
            logging.warn("Did not found the phase {}! Try using offline or online phase!".format(experiment_phase))

    @staticmethod
    def import_data_for_phase(file_name, ex_phase=ONLINE_PHASE, dataset_type=SYNTHETIC):
        if dataset_type == REAL_WORLD:
            return FileImporter.import_online_dataset(filepath=FileImporter.REAL_WORLD_DATA_PATH, filename=file_name)
        if ex_phase == Constants.ONLINE_PHASE:
            return FileImporter.import_online_dataset(file_name)
        elif ex_phase == Constants.OFFLINE_PHASE:
            return FileImporter.import_offline_data(file_name)
        else:
            logging.warn("Did not found the phase {}! Try using offline or online phase!".format(ex_phase))

    @staticmethod
    def import_external_metric_optimizer_merged_result(init_strategy, experiment):
        return FileImporter.import_data_as_df(filename=optimizers_merged_filename_external,
                                              filepath=optimizer_export_path.format(experiment=experiment,
                                                                                    warmOrColdstart=init_strategy))

    @staticmethod
    def import_optimizer_merged_result(init_strategy, experiment):
        return FileImporter.import_data_as_df(filename=optimizers_merged_filename,
                                              filepath=optimizer_export_path.format(experiment=experiment,
                                                                                    warmOrColdstart=init_strategy))


def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


class FileExporter:
    export_path = PROJECT_PATH + "/result/kmeanstraining/synthetic/benchmark_"

    @staticmethod
    def file_exists(path, filename):
        """
        Checks if the file with 'filename' already exists in the given 'path'.
        Returns True if the file already exists in this path and else False.
        """
        return os.path.exists(os.path.join(path, filename))

    @staticmethod
    def export_dataframe_to_csv(results_df, path, file_name, sep=DEFAULT_DELIMITER, decimal=DEFAULT_DECIMAL):
        """
        Exports a dataframe, that is, e.g., the result of an experiment, to a csv file.
        This ensures that the same decimal, separation and float format is used for all experiments.
        """
        create_path_if_not_exists(path)
        full_result_path = os.path.join(path, file_name)
        results_df.to_csv(path_or_buf=full_result_path, sep=sep, index=False, decimal=decimal, float_format="%.5f")

    @staticmethod
    def export_dict_to_csv(dic, path, file_name):
        """
        Utility method for also exporting a dictionary to a csv file.
        Actually converts the dic to a dataframe and then uses the method 'export_dataframe_to_csv'.
        """
        create_path_if_not_exists(path)
        df = pd.DataFrame.from_dict(data=dic)
        FileExporter.export_dataframe_to_csv(results_df=df, path=path, file_name=file_name)

    @staticmethod
    def save_model(model, filepath, filename):
        create_path_if_not_exists(filepath)
        joblib.dump(model, os.path.join(filepath, filename))

    @staticmethod
    def get_results_path(optimizer_name, experiment, metric_name, repetition, init_strategy, ex_phase, t=1,
                         n_warmstarts=None, dataset_type=REAL_WORLD):
        if ex_phase == Constants.OFFLINE_PHASE:
            return FileImporter.get_offline_optimizer_results_path(optimizer_name, metric_name, experiment)
        elif ex_phase == Constants.ONLINE_PHASE:
            return FileImporter.get_online_optimizer_results_path(optimizer_name, metric_name, repetition, experiment,
                                                                  init_strategy, dataset_type, t=t,
                                                                  n_warmstarts=n_warmstarts)

    @staticmethod
    def export_meta_features_ferrari(dic, file_name):
        if isinstance(dic, dict):
            df = pd.DataFrame.from_dict(data=dic)
        else:
            df = dic
        FileExporter.export_dataframe_to_csv(df, path=META_FEATURE_PATH_FERRARI, file_name=file_name)

    @staticmethod
    def ferrari_already_exists(dataset):
        return FileExporter.file_exists(META_FEATURE_PATH_FERRARI, dataset)


def get_filename_from_path(file_name_with_path):
    """
    Given a path that also contains the file name, this method return the filename form the path.
    """
    head, tail = os.path.split(file_name_with_path)
    return tail


def get_dir_from_path(file_path):
    """
    Given a path that also contains the file name, this method return the path without the filename.
    """
    head, tail = os.path.split(file_path)
    return head


def extract_k_from_filename(file):
    """
    Method to extract the actual number of clusters of a dataset from the filename.
    Note that this is only possible for the datasets used in the experiments (more specifically in the online_dataset and
    offline_dataset folder) since they follow a specific naming convention.
    """
    digits = list(map(int, re.findall('\\d+', file)))
    # k value is the third digit
    return digits[2]


def get_dataset_information(file):
    """
    Returns a tuple with informations of the dataset.
    Note that this is only possible for the datasets used in the experiments (more specifically in the online_dataset and
    offline_dataset folder) since they follow a specific naming convention.
    :return: Tuple object (n, d, k, r) where n is the number of instances, d the number of features,
     k the actual number of clusters and r the noise of the dataset.
    """
    digits = list(map(int, re.findall('\\d+', file)))
    print("digits: {}".format(digits))
    nr_instances = digits[0]
    nr_features = digits[1]
    k = digits[2]
    noise = digits[3] / 100
    return nr_instances, nr_features, k, noise


if __name__ == "__main__":
    real_world_datasets = FileImporter.list_data_for_phase(ONLINE_PHASE, dataset_type=REAL_WORLD)
    #dic = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "W": 9, "X": 10, "Y": 11}
    for real_world_dataset in real_world_datasets:
        if not "Epileptic"  in real_world_dataset:
            continue
        raw_data = pd.read_csv(FileImporter.REAL_WORLD_DATA_PATH + real_world_dataset, header=None, sep=";")
        print(raw_data)
        raw_data.to_csv(path_or_buf=FileImporter.REAL_WORLD_DATA_PATH+ real_world_dataset, index=False, header=None, sep=",")

"""
        avila_dataset, true_labels = FileImporter.import_data(FileImporter.REAL_WORLD_DATA_PATH + real_world_dataset)
        print(avila_dataset.info())
        # avila_dataset.reset_index(inplace=True, drop=True)
        n, d, k, r = get_dataset_information(real_world_dataset)
        print("n: {}, d:{}, k: {}, r: {}".format(n, d, k, r))

        # avila_dataset = avila_dataset.apply(lambda x: pd.to_numeric(x, errors='coerce'))
        # avila_dataset = avila_dataset.fillna(avila_dataset.mean())
        # if "Motion" in real_world_dataset:
        #    avila_dataset = avila_dataset.drop([33, 34, 35], axis=1)
        # print(avila_dataset.info())
        print("Running kmeasn on {}".format(real_world_dataset))
        clustering_result = ClusteringAlgorithms.run_kmeans(avila_dataset, k=k)
        print("Finished kmeans on {}".format(real_world_dataset))
        labels_predicted = clustering_result.labels
        score = MetricCollection.ADJUSTED_MUTUAL.score_metric(avila_dataset, labels=labels_predicted, true_labels=true_labels)
        print("Predicted labels: {}".format(labels_predicted))
        print("true labels: {}".format(true_labels))
        print("AMI score: {}".format(score))
        avila_dataset.to_csv(path_or_buf=FileImporter.REAL_WORLD_DATA_PATH+ real_world_dataset, index=False, header=None, sep=",")
"""