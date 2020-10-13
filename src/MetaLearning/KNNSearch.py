import logging

import numpy as np
from sklearn.externals import joblib
from sklearn.neighbors.kd_tree import KDTree

from MetaLearning import MetaFeatureExtractor
from Utils.Constants import SYNTHETIC, REAL_WORLD, ONLINE_PHASE
from Utils.FileUtil import FileExporter, FileImporter, PROJECT_RESULTS_PATH
import pandas as pd
from Utils import FileUtil

META_FEATURE_PATH = MetaFeatureExtractor.OUTPUT_PATH
META_FEATURE_INDEX_PATH = META_FEATURE_PATH + "/indices"
KDTREE_FILENAME = "kdtree.pkl"
KDTREE_PATH = META_FEATURE_PATH + "/kdtree"
REAL_WORLD_KDTREE_PATH = PROJECT_RESULTS_PATH + "/" + REAL_WORLD + "/metafeatures/"


def build_train_kd_tree(dataset_type=SYNTHETIC):
    kd_tree_path = KDTREE_PATH
    if dataset_type == REAL_WORLD:
        kd_tree_path = REAL_WORLD_KDTREE_PATH
    datasets = FileImporter.list_online_data(dataset_type=dataset_type)
    meta_feature_list = []
    dataset_knn_df = pd.DataFrame(columns=["knn_id", "name"])

    for index, file in enumerate(datasets):
        print("get metafeatures from dataset {}".format(file))

        # Also save index of the meta_features to the corresponding dataset.
        # Is needed because kd-tree only gives the index and we can only save the meta-features
        dataset_knn_df = dataset_knn_df.append({"knn_id": index, "name": file}, ignore_index=True)
        mf_dic = MetaFeatureExtractor.extract_meta_features_as_dic(file, phase=ONLINE_PHASE, dataset_type=dataset_type)
        meta_feature_values = [mf_dic[key] for key in sorted(mf_dic.keys())]
        meta_feature_list.append(meta_feature_values)

        print("Imported MetaLearning from file: {}".format(file))

    # DBService.commit_changes()
    print("Constructing kd-tree")
    kdt = KDTree(np.array(meta_feature_list), metric='manhattan')
    print("Finished constructing kdtree")
    print("Saving kdtree model to file")
    FileExporter.save_model(model=kdt, filepath=kd_tree_path, filename=KDTREE_FILENAME)
    print("Finished saving kdtree model")
    FileExporter.export_dataframe_to_csv(results_df=dataset_knn_df, file_name="datasets_knn_id.csv", path=kd_tree_path)
    # print("saving kdtree model file in db")
    # DBService.save_file_path(KDTREE_PATH, KDTREE_FILENAME)


def find_nearest_neighbors(online_metafeature_array, k=1, dataset_type=SYNTHETIC):
    # first load the kd-tree model
    # knn_search = DBService.get_nearest_neighbor_file()
    # print(knn_search)
    kd_path = KDTREE_PATH

    logging.info("dataset_type: {}".format(dataset_type))
    nn_datasets_df = pd.read_csv(FileUtil.META_FEATURE_PATH + "/datasets_knn_id.csv", sep=';')
    logging.info("nn datasets df: {}".format(nn_datasets_df))

    kdtree_model = joblib.load(kd_path + '/' + KDTREE_FILENAME)
    print(online_metafeature_array)
    if dataset_type == REAL_WORLD:
        dist, ind = kdtree_model.query(online_metafeature_array, k=k + 1)
    else:
        dist, ind = kdtree_model.query(online_metafeature_array, k=k)
    print("Index: {}".format(ind))
    nn_datasets = []
    for index in ind:
        if dataset_type == REAL_WORLD:
            # FOr real-world datasets take the second best, because best is the same dataset
            nn_dataset_for_ind = nn_datasets_df[nn_datasets_df["knn_id"] == index[1]]
        else:
            nn_dataset_for_ind = nn_datasets_df[nn_datasets_df["knn_id"] == index[0]]
        nn_dataset_name = nn_dataset_for_ind["name"].values[0]
        nn_datasets.append(nn_dataset_name)
    print("nn datasets: ".format(nn_datasets))
    return nn_datasets
