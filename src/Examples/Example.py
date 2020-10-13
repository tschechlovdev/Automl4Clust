from sklearn.datasets import make_blobs
import ConfigSpace as CS

from Algorithm import ClusteringAlgorithms
from Metrics.MetricHandler import MetricCollection
from Optimizer.Optimizer import SMACOptimizer, RandomOptimizer, BOHBOptimizer, HyperBandOptimizer
import ConfigSpace.hyperparameters as CSH

# Create a testing data set for all examples
X, y = make_blobs(n_samples=10000, n_features=10)

# optimizers that can be used in our implementation
optimizers = [RandomOptimizer, SMACOptimizer, HyperBandOptimizer, BOHBOptimizer]

##########################################################
### Example 1: Running optimizer with default settings ###
# We use Hyperband in our examples
optimizer = HyperBandOptimizer

# Simple example running optimizer with default settings
automl_four_clust_instance = optimizer(dataset=X)
result = automl_four_clust_instance.optimize()
best_configuration = automl_four_clust_instance.get_best_configuration()

##################################################################
### Example 2: Running optimizer with custom budget and metric ###
# Running optimizer with custom budget and metric
automl_four_clust_instance = optimizer(dataset=X, metric=MetricCollection.DAVIES_BOULDIN, n_loops=20)
result = automl_four_clust_instance.optimize()
best_configuration = automl_four_clust_instance.get_best_configuration()

# It is also possible to get the history of configurations
history = automl_four_clust_instance.get_config_history()

######################################################
### Example 3: Running optimizer with warmstarting ###

# We pass the warmstart configurations as list of dictionaries with the hyperparameter "k" and the name of the algorithm
warmstart_configs = [{"k": 2, "algorithm": ClusteringAlgorithms.KMEANS_ALGORITHM},
                     {"k": 10, "algorithm": ClusteringAlgorithms.GMM_ALGORITHM}]

# Run optimizer with the warmstart configurations
automl_four_clust_instance = optimizer(dataset=X, warmstart_configs=warmstart_configs)
result = automl_four_clust_instance.optimize()
best_configuration = automl_four_clust_instance.get_best_configuration()

####################################################################
### Example 4: Running optimizer with custom configuration space ###

# define a custom k_range
k_range = (2, 200)
# define algorithms
algorithms = [ClusteringAlgorithms.GMM_ALGORITHM, ClusteringAlgorithms.KMEDOIDS]

# Define configuraton space object based on the k_range and the algorithms
cs = CS.ConfigurationSpace()
algorithm_hyperparameter = CSH.CategoricalHyperparameter("algorithm", choices=algorithms)
cs.add_hyperparameter(algorithm_hyperparameter)
k_hyperparameter = CSH.UniformIntegerHyperparameter("k", lower=k_range[0],
                                                    upper=k_range[1])

automl_four_clust_instance = optimizer(dataset=X, cs=cs)
result = automl_four_clust_instance.optimize()
best_configuration = automl_four_clust_instance.get_best_configuration()
