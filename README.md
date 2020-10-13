# AutoML4Clust

Repository for the prototypical implementation of AutoML4Clust.
It implements implements different instantiations of our proposed AutoML4Clust.
To this end, we use different state-of-the art optimizers from existing AutoML systems and apply them to the unsupervised task of clustering.
Furthermore, we implemented meta-learning for the unsupervised task of clustering to warmstart the optimizers.
In this prototype, we focus on k-center algorithms.
Furthermore, other clustering algorithms, e.g., from other clustering families, can be seamlessly added.

### Prerequisites

To install the AutoML4Clust API, you require Python 3.6 and a Linux environment with Ubuntu >= 16.04.
You can find and install the required libraries from the `requirements.txt` file.
 
### Optimizers
We used four state-of-the-art optimizers from existing AutoML systems with the following implementations:
- Random Search: [scikit-optimize](https://scikit-optimize.github.io/stable/)
- Bayes optimization with Random Forests: [SMAC](https://github.com/automl/SMAC3)
- Hyperband and BOHB: [hpbandster](https://github.com/automl/HpBandSter)

For warmstarting Hyperband and BOHB we used the code provided by the BOHB authors [here](https://github.com/automl/HpBandSter/issues/71). 

### Clustering algorithms and metrics

For the clustering algorithms and metrics, we rely on the prominent ml-library [scikit-learn](https://scikit-learn.org/stable/modules/clustering.html).
To this end, we used the k-Means, MiniBatchK-Means, GMM and k-Medoids algorithms, which rely on different objective functions, thus achieving different clustering results.
For the k-Medoids algorithm, we used [scikit-learn extra](https://scikit-learn-extra.readthedocs.io/en/latest/generated/sklearn_extra.cluster.KMedoids.html)
implementation.

We also used the three internal metrics that are implemented in scikit-learn for clustering, i.e.,
the Calinski-Harabasz, Davies-Bouldin Index and the Silhouette.

### Simple API Example

The API is very simple to use.
In the following, we show examples on how to run the API on a synthetically generated dataset.
Per default, our API uses the Calinski-Harabasz metric, a budget of `n_loops=60`, the four above-mentioned clustering algorithms 
and a range for the hyperparameter k of `(2, n/10)`, where `n` is the number of entities in the dataset.
However, in the `Examples` directory you can find examples on how to use another (i) configuration space, 
(ii) clustering metric, (iii) budget, and (iv) how to use warmstarting.

````python
from sklearn.datasets import make_blobs

from Optimizer.Optimizer import SMACOptimizer, RandomOptimizer, BOHBOptimizer, HyperBandOptimizer

# Create a testing data set for all examples
X, y = make_blobs(n_samples=1000, n_features=10)

# optimizers that can be used in our implementation
optimizers = [RandomOptimizer, SMACOptimizer, HyperBandOptimizer, BOHBOptimizer]

# We use Hyperband in our examples
optimizer = HyperBandOptimizer

# Instantiating AutoML4Clust on the dataset and getting the best found configuration
automl_four_clust_instance = optimizer(dataset=X)
result = automl_four_clust_instance.optimize()
best_configuration = automl_four_clust_instance.get_best_configuration()
````

Since the API is simple to use, it can be easily integrated into existing analysis pipelines.
