"""
Contains the code that is responsible for extracting the meta-features for clustering.
It basically uses the pymfe package and extracts all meta-features that do not require class labels.
"""
import logging
import math
import sys

from pymfe.mfe import MFE
import numpy as np
import pandas as pd

from Utils import FileUtil, Constants
from Utils.Constants import SYNTHETIC, REAL_WORLD

try:
    from Utils.FileUtil import FileImporter, FileExporter
    from MetaLearning import MetaFeatures

except ModuleNotFoundError:
    # strange workaround because Metrics module cannot be found
    sys.path.append("/home/ubuntu/automlclustering")
    from Utils.FileUtil import FileImporter, FileExporter
    from MetaLearning import MetaFeatures

OUTPUT_PATH = FileUtil.PROJECT_PATH + "/results/result/metafeatures/"
logging.basicConfig(filename='metafeatures.log',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s: %(message)s')


def extract_meta_features_as_dic(file_name=None, data=None, data_labels=None, phase=Constants.OFFLINE_PHASE,
                                 dataset_type=SYNTHETIC):
    if file_name:
        data, data_labels = FileImporter.import_data_for_phase(file_name, ex_phase=phase, dataset_type=dataset_type)
        print(data.info())
        data = np.array(data)
        print(data)

    # Extract general, statistical and information-theoretic measures
    mfe = MFE(features=MetaFeatures.META_FEATURE_LIST, summary=MetaFeatures.SUMMARY_LIST)
    # Actualy we do not need data_labels, but it HAS to be passed to fit()
    mfe.fit(data, data_labels, transform_num=False)

    att, val = mfe.extract()
    dic = {}
    val_without_nan = [x for x in val if not math.isnan(x)]
    avg_value = sum(val_without_nan) / len(val_without_nan)

    for a, v in zip(att, val):
        dic[a] = avg_value if math.isnan(v) else v
    logging.info(str(dic))
    return dic


def extract_meta_features_as_df(file_name, file_path=None, dataset_type=SYNTHETIC, phase=Constants.ONLINE_PHASE):
    dic = extract_meta_features_as_dic(file_name, file_path, dataset_type=dataset_type, phase=phase)
    # values have to be a list for pandas df
    dic = {k: [v] for k, v in dic.items()}
    return pd.DataFrame.from_dict(data=dic)


def save_meta_features_to_file():
    file_names = FileImporter.get_offline_file_list()
    for file_name in file_names:
        df = extract_meta_features_as_df(file_name)
        FileExporter.export_dataframe_to_csv(results_df=df, file_name=file_name, path=OUTPUT_PATH)


if __name__ == '__main__':
    for file in FileImporter.list_online_data(dataset_type=REAL_WORLD):
        print("Extracting meatafeatures for {}".format(file))
        real_world_data = FileImporter.import_data_for_phase(file, dataset_type=REAL_WORLD)
        mf_df = extract_meta_features_as_df(file, dataset_type=REAL_WORLD)
        print("Extracted metafeautres")
        print("Saving result to {}/{}".format(FileUtil.META_FEATURE_PATH_REAL_WORLD, file))
        FileExporter.export_dataframe_to_csv(mf_df, path=FileUtil.META_FEATURE_PATH_REAL_WORLD, file_name=file)
