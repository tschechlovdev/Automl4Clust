from datetime import datetime

from pyclustering.cluster.encoder import type_encoding, cluster_encoder
from pyclustering.cluster.gmeans import gmeans
from pyclustering.cluster.xmeans import xmeans
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
import logging
import time

from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
from sklearn_extra.cluster import KMedoids

default_max_iterations = 10
KMEANS_ALGORITHM = "KMeans"
GMM_ALGORITHM = "GMM"
KMEDOIDS = "KMedoids"
MINI_BATCH_KMEANS = "MBKMeans"
algorithms = [KMEANS_ALGORITHM, GMM_ALGORITHM, KMEDOIDS, MINI_BATCH_KMEANS]

logging.basicConfig(level=logging.ERROR)


class ClusteringResult:
    """
    Model for one Clustering Algorithm result for one k.
    Contains the no of iterations, execution time, the sse and the calculated labels.
    """
    def __init__(self, k, iterations, labels, sse=None, execution_time=None):
        self.execution_time = execution_time
        self.k = k
        self.iterations = iterations
        self.sse = sse
        self.labels = labels


def run_kmeans(data_without_labels, k, max_iterations=default_max_iterations, kmeans_algo=KMeans,
               algo_name=KMEANS_ALGORITHM):
    """
    Runs kmeans with default settings.
    Gets the dataset and the number of clusters (k) as argument and returns a :pyclass:KmeansResult object.
    Optionally the number of max iterations can be passed, otherwise the default value will be used.
    :param max_iterations: int, number of max_iterations, the default is used if not specified
    :param data_without_labels: np.array containing the dataset without labels
    :param k: int, number of clusters
    :param kmeans_algo: Kmeans algorithm to run. Can be for example the KMeans or the MiniBatchKMeans algorithm
    :return: :pyclass:KmeansResult
    """
    logging.info("{Timestamp}: Running {algo} with k={k} and maxIter={maxIter}".format(Timestamp=datetime.now(), k=k,
                                                                                       maxIter=max_iterations,
                                                                                       algo=algo_name))
    # random_state = np.random.RandomState(0)
    # print(random_state.randint(0, 42))
    start_time = time.time()
    kmeans = kmeans_algo(n_clusters=k, max_iter=max_iterations, n_init=1).fit(X=data_without_labels)
    execution_time = time.time() - start_time
    logging.info("{Timestamp}: {algo} took {execution_time}s".format(Timestamp=datetime.now(),
                                                                     execution_time=execution_time, algo=algo_name))
    return ClusteringResult(k=k, iterations=kmeans.n_iter_, sse=kmeans.inertia_,
                            labels=kmeans.labels_, execution_time=execution_time)


def run_gmm(data_without_labels, k, max_iterations=default_max_iterations, init="kmeans"):
    gmm = GaussianMixture(k, max_iter=max_iterations, init_params=init)
    labels = gmm.fit_predict(data_without_labels)
    return ClusteringResult(k=k, iterations=gmm.n_iter_, labels=labels)


def run_ward(data_without_labels, k, full_tree=False):
    labels = AgglomerativeClustering(linkage='ward', n_clusters=k, compute_full_tree=full_tree).fit_predict(
        data_without_labels)
    # number of iterations unknown for hierarchical clustering
    return ClusteringResult(k=k, labels=labels, iterations=-1)


def run_mini_batch_kmeans(data_without_labels, k, max_iterations=default_max_iterations):
    return run_kmeans(data_without_labels, k, max_iterations, MiniBatchKMeans, MINI_BATCH_KMEANS)


def run_kmedoids(data_without_labels, k, max_iterations=default_max_iterations):

    try:
        k_medoids = KMedoids(n_clusters=k, max_iter=max_iterations)
        labels = k_medoids.fit_predict(data_without_labels)
        # kmedoids can produce Memory Errors, a fix for this is provided here:
        # https://stackoverflow.com/questions/57507832/unable-to-allocate-array-with-shape-and-data-type
        # However leave this for now to be on the safe side.
    except MemoryError:
        logging.error("Memory error for dataset with n={n} and d={}. "
                      "Returning default Clustering Result with labels = [-1 ... -1]")
        return ClusteringResult(k=k, labels=[-1 for x in data_without_labels], iterations=1)
    # have to transform clusters (that contain the indexes of points that belong to it) into list of labels
    return ClusteringResult(k=k, labels=labels, iterations=k_medoids.n_iter_)

def run_algorithm( algorithm_name, data_without_labels, k, max_iterations=default_max_iterations):
    """
    Runs a clustering algorithm based on the name.
    Therefore, it executes the run_X function for the corresponding algorithm.
    The currently supported algorithms are k-Means, k-Medoids, MiniBatchK-Means and Gaussian Mixture Models.

    :param algorithm_name: Name of the algorithm
    :param data_without_labels: np.array that contains the dataset, without actual labels of the dataste
    :param k: Number of clusters to use for the algorithms.
    :param max_iterations: Number of iterations to perform.
    :return: instantiation of a ClusteringResult
    """

    logging.info("Starting {} with k={} and maxiterations={}".format(algorithm_name, k, max_iterations))
    start_time = time.time()
    clustering_result = ClusteringResult(k=-1, labels=None, iterations=-1)
    if algorithm_name == GMM_ALGORITHM:
        clustering_result = run_gmm(data_without_labels, k, max_iterations)
    elif algorithm_name == KMEANS_ALGORITHM:
        clustering_result = run_kmeans(data_without_labels, k, max_iterations)
    elif algorithm_name == KMEDOIDS:
        clustering_result = run_kmedoids(data_without_labels, k, max_iterations)
    elif algorithm_name == MINI_BATCH_KMEANS:
        clustering_result = run_kmeans(data_without_labels, k, max_iterations, MiniBatchKMeans, MINI_BATCH_KMEANS)
    else:
        logging.warn(
            "unknown algorithm {} found - The algorithm is currently not supported! No algorithm is executed".format(
                algorithm_name))

    algorithm_time = time.time() - start_time
    clustering_result.execution_time = algorithm_time
    logging.info("Finished calculating {}, took {}s and {} iterations"
                 .format(algorithm_name, clustering_result.execution_time, clustering_result.iterations))
    return clustering_result




#########################################################################################################
######## GMeans and XMeans implementations using pyclustering ###########################################
#########################################################################################################

def run_gmeans_pyclustering(data, k_max=200):
    #k_max = int(len(data)/10)
    gmeans_instance = gmeans(data, k_max=k_max)
    gmeans_instance.process()
    clusters = gmeans_instance.get_clusters()

    encoder = cluster_encoder(type_encoding.CLUSTER_INDEX_LIST_SEPARATION, clusters, data)
    encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
    labels = encoder.get_clusters()
    return labels


def run_xmeans(data, k_max=200):
    xmeans_instance = xmeans(data, kmax=k_max, ccore=False)
    xmeans_instance.process()
    clusters = xmeans_instance.get_clusters()

    encoder = cluster_encoder(type_encoding.CLUSTER_INDEX_LIST_SEPARATION, clusters, data)
    encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
    labels = encoder.get_clusters()

    return labels

if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)

    k_values = [5, 10, 15, 20]
    X, y = make_blobs(n_samples=1000, n_features=10, centers=5)
    print("running with max_iterations = 1:")
    for k in k_values:
        #   for tol in tolerances:
        start = time.time()
        result = run_gmm(X, k, max_iterations=1)
        end = time.time() - start
        print("GMM with k={} and took: {}s".format(k, end))

    print("----------------------------------")
    print("running with max_iterations = 2:")
    for k in k_values:
        #   for tol in tolerances:
        start = time.time()
        result = run_gmm(X, k, max_iterations=2)
        end = time.time() - start
        print("GMM with k={} and took: {}s".format(k, end))

    print("----------------------------------")
    print("running with max_iterations = 3:")
    for k in k_values:
        #   for tol in tolerances:
        start = time.time()
        result = run_gmm(X, k, max_iterations=3)
        end = time.time() - start
        print("GMM with k={} and took: {}s".format(k, end))


    print("----------------------------------")
    print("running with max_iterations = 10:")
    for k in k_values:
        #   for tol in tolerances:
        start = time.time()
        result = run_gmm(X, k)
        end = time.time() - start
        print("GMM with k={} and took: {}s".format(k, end))
