from sklearn import metrics
import time
import logging
import math
import numpy as np

"""
Responsible for everything related to Metrics.
It contains all metrics, the metricresult class, the collection of metrics that are used, the metric class itself 
and the MetricEvaluator.
"""


class MetricResult:
    """
        Class that describes the information that is saved for each metric after calculating the metric result for a
        given kmeans result. Is used to represent the result of the MetricEvaluator.run_metrics() method.
    """

    def __init__(self, execution_time, score, name, metric_type):
        self.execution_time = execution_time
        self.score = score
        self.name = name
        self.metric_type = metric_type


class MetricType:
    EXTERNAL = "External"
    INTERNAL = "Internal"


class Metric:
    """
        Basic entity that describes a metric. For each metric there is one instance of this class which will be saved in
        the MetricCollection class.
        This class is also responsible for evaluating the metric score in a generic way.
    """

    def __init__(self, name, score_function, metric_type, sample_size=None):
        self.name = name
        self.score_function = score_function
        self.metric_type = metric_type
        self.sample_size = sample_size

    def get_abbrev(self):
        return MetricCollection.METRIC_ABBREVIATIONS[self.name]

    def score_metric(self, data, labels=None, true_labels=None):
        """
        Calculates the score of a metric for a given dataset and the corresponding class labels. If the metric is an
        external metric, also the true_labels have to be passed to calculate the metric. :param data: the raw dataset
        without labels
        :param data:
        :param labels: the labels that were calculated for example by kmeans
        :param true_labels: the gold standard labels of the dataset (is needed for external metrics)
        :return: the result of the metric calculation, which should be a float. It is the negative value of a metric if
        the metric should be optimized(since we want to minimize the value)
        """
        if labels is None and true_labels is None:
            logging.error("either labels or true_labels has to be set to get metric score")
            raise Exception("either labels or true_labels has to be set to get metric score")

        logging.info("Start scoring metric {}".format(self.name))

        # if sample size is defined, then we want to score metric, but with given sample size (only for silhouette)
        if self.name == MetricCollection.SILHOUETTE_SAMPLE_10.name:
            n = data.shape[0]
            if self.sample_size:
                n_sample_size = int(n * self.sample_size)
                n_clusters = len(np.unique(labels))
                # if number of samples is lower than labels we just return 1
                if n_sample_size < n_clusters:
                    logging.warning("sample size {} is lower than number of clusters {}, so setting score to 1"
                                    .format(n_sample_size, n_clusters))
                    score = 1
                else:
                    # Multiply with -1 because we want to minimize
                    score = -1 * self.score_function(data, labels, sample_size=n_sample_size)


        # if internal just calculate score by data and labels
        elif self.metric_type == MetricType.INTERNAL:
            score = self.score_function(data, labels)
            if self.name == MetricCollection.CALINSKI_HARABASZ.name or self.name== MetricCollection.SILHOUETTE.name:
                # If Calinski-Harabasz or Silhouette then we have to multiply by -1 because should be maximized
                score = -1 * score

        # if external metric then we need the "ground truth" instead of the data
        elif self.metric_type == MetricType.EXTERNAL:
            score = -1 * self.score_function(true_labels, labels)

        else:
            logging.error("There was an unknown metric type which couldn't be calculated. The metric is " + self.name)
            score = math.inf

        logging.info("Scored metric {} and value is {}".format(self.name, score))
        return score


class MetricCollection:
    """
        Contains all metrics that are used for the experiments. The metrics can be get by either calling all_metrics or
        using the get_all_metrics_sorted method.
    """

    SILHOUETTE_SAMPLE = "Sampled Silhouette"
    SILHOUETTE = Metric("Silhouette", metrics.silhouette_score, MetricType.INTERNAL)
    CALINSKI_HARABASZ = Metric("Calinski-Harabasz", metrics.calinski_harabasz_score, MetricType.INTERNAL)
    DAVIES_BOULDIN = Metric("Davies-Bouldin", metrics.davies_bouldin_score, MetricType.INTERNAL)
    ADJUSTED_RAND = Metric("Adjusted Rand", metrics.adjusted_rand_score, MetricType.EXTERNAL)
    ADJUSTED_MUTUAL = Metric("Adjusted Mutual", metrics.adjusted_mutual_info_score, MetricType.EXTERNAL)
    HOMOGENEITY = Metric("Homogeneity", metrics.homogeneity_score, MetricType.EXTERNAL)
    V_MEASURE = Metric("V-measure", metrics.v_measure_score, MetricType.EXTERNAL)
    COMPLETENESS_SCORE = Metric("Completeness", metrics.completeness_score, MetricType.EXTERNAL)
    FOWLKES_MALLOWS = Metric("Folkes-Mallows", metrics.fowlkes_mallows_score, MetricType.EXTERNAL)

    # abbreviations are useful for, e.g., plots
    METRIC_ABBREVIATIONS = {
        SILHOUETTE.name: "SIL",
        CALINSKI_HARABASZ.name: "CH",
        DAVIES_BOULDIN.name: "DBI",
        ADJUSTED_RAND.name: "AR",
        ADJUSTED_MUTUAL.name: "AMI",
        HOMOGENEITY.name: "HG",
        V_MEASURE.name: "VM",
        COMPLETENESS_SCORE.name: "CS",
        FOWLKES_MALLOWS.name: "FM"
    }
    internal_metrics = [CALINSKI_HARABASZ, DAVIES_BOULDIN,
                        #SILHOUETTE_SAMPLE_10,
                        SILHOUETTE]
    external_metrics = [ADJUSTED_MUTUAL, ADJUSTED_RAND,
                        COMPLETENESS_SCORE,
                        FOWLKES_MALLOWS,
                        HOMOGENEITY, V_MEASURE]
    all_metrics = external_metrics + internal_metrics
    experiment_metrics = [CALINSKI_HARABASZ, DAVIES_BOULDIN,
                          #SILHOUETTE,
                          ADJUSTED_MUTUAL]

    @staticmethod
    def get_metric_by_abbrev(metric_abbrev):
        print(metric_abbrev)
        for metric in MetricCollection.all_metrics:
            if MetricCollection.METRIC_ABBREVIATIONS[metric.name] == metric_abbrev:
                return metric

    @staticmethod
    def get_all_metrics_sorted_by_name():
        """
        Returns all metrics in sorted order. This is important, if e.g. calculations were done and you want to map
        value to their corresponding name.
        :return:
        """
        MetricCollection.all_metrics.sort(key=lambda x: x.name)
        return MetricCollection.all_metrics

    @staticmethod
    def get_sorted_abbreviations_by_type():
        return [MetricCollection.METRIC_ABBREVIATIONS[metric.name] for metric
                         in MetricCollection.all_metrics]

    @staticmethod
    def get_sorted_abbreviations_internal_by_type():
        return [MetricCollection.METRIC_ABBREVIATIONS[metric.name] for metric
                         in MetricCollection.internal_metrics]

    @staticmethod
    def get_abrev_for_metric(metric_name):
        return MetricCollection.METRIC_ABBREVIATIONS[metric_name]


class MetricEvaluator:

    @staticmethod
    def run_metrics(data, true_labels, labels):
        """
        :param data: dataset that is the raw dataset and without labels
        :param true_labels: the labels of the ground truth
        :param labels: predicted labels that were found by the clustering algorithm
        :return: List of :py:class:`MetricResult`, one for each Metric that was used.
        """
        logging.info("start calculating metrics")
        result = []

        for metric in MetricCollection.all_metrics:
            metric_name = metric.name
            logging.info("start calculation for " + metric_name)
            metric_execution_start = time.time()
            score = metric.score_metric(data, labels=labels, true_labels=true_labels)
            metric_execution_time = time.time() - metric_execution_start
            metric_result = MetricResult(name=metric.name, score=score, execution_time=metric_execution_time,
                                         metric_type=metric.metric_type)
            logging.info("Finished {} with score {} and execution time {}"
                         .format(metric_name, score, metric_execution_time))
            result.append(metric_result)
        return result
