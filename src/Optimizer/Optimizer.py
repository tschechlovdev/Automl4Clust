"""
Implements the optimizers that are used for hyperparameter-tuning.
The implemented algorithms are the bayesian optimization, random search,
hyperband and a combination of bayesian optimization and hyperband (BOHB)
"""
import json
import logging
import random
import time
from math import inf

import pandas as pd
import numpy as np
from ConfigSpace.configuration_space import Configuration
from hpbandster.core.base_iteration import BaseIteration
from hpbandster.core.master import Master
from hpbandster.core.result import Result
from hpbandster.optimizers.config_generators.bohb import BOHB as BOHB_generator
from hpbandster.optimizers.config_generators.random_sampling import RandomSampling
import uuid
from abc import abstractmethod, ABC

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB as BOHB, HyperBand
from scipy.optimize import OptimizeResult
from sklearn.datasets import make_blobs
from skopt import gp_minimize, dummy_minimize, BayesSearchCV, Optimizer
from skopt.space import Integer, Categorical
from smac.facade.func_facade import fmin_smac
from smac.facade.smac_ac_facade import SMAC4AC
from smac.facade.smac_bo_facade import SMAC4BO
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.initial_design.initial_design import InitialDesign
from smac.initial_design.sobol_design import SobolDesign
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae.execute_ta_run import ExecuteTARun

from Algorithm import ClusteringAlgorithms
from Algorithm.ClusteringAlgorithms import run_kmeans, run_algorithm
from Metrics.MetricHandler import MetricCollection
from Utils import Helper


class AbstractOptimizer(ABC):
    """
       Abstract Wrapper class for all implemented optimization methods.
       Basic purpose is to have convenient way of using the optimizers.
       Therefore, after initializing the optimizer, just by running the
       optimize function the best result found by the optimizer will be obtained.
    """
    n_loops = 60

    def __init__(self, dataset, metric=MetricCollection.CALINSKI_HARABASZ, search_space=None, true_labels=None,
                 warmstart_configs=None,
                 file_name=None, rep=0, cs=None, n_loops=60):
        """

        :param dataset: np.array of the dataset (without the labels)
        :param metric: A metric from the MetricCollection. Default is CALINSKI_HARABASZ
        :param cs: ConfigurationSpace object that is used by the optimizer. If not passed, then default is used.
        :param true_labels: true labels of the dataset. This can be passed but is not by the used. Can be useful for evaluation purposes.
        :param warmstart_configs: Warmstart configurations. Has to be a list of parameter list, e.g., [2, KMEANS_ALGORITHM].
        :param file_name: file_name of the dataset can be used to reference the dataset with its name.
        :param rep: If repeatedly executed the same optimizer, a repetition number can be given for reference.
        """
        if not cs and not search_space:
            logging.warning(
                "You should either pass a ConfigSpace object or a serch space as list of the parameter ranges. "
                "Using default search space.")
        if not metric:
            self.metric = MetricCollection.CALINSKI_HARABASZ
        else:
            self.metric = metric
        self.dataset = dataset
        self.true_labels = true_labels
        self.metric_runtimes = []

        self.warmstart_configs = warmstart_configs
        self.algo_runtimes = []
        self.optimizer_runtimes = []
        self.start_time = None
        self.file_name = file_name
        self.rep = rep
        self.cs = cs
        self.n_loops = n_loops
        if not cs:
            self.cs = self.get_configspace()
        else:
            self.cs = cs

        # define search space from cs -> Is used by random optimizer since it has a different API
        lower = self.cs.get_hyperparameter("k").lower
        upper = self.cs.get_hyperparameter("k").upper
        k_range = (lower, upper)
        print("k range: {}".format(k_range))
        self.search_space = [k_range]
        if "algorithm" in self.cs.get_hyperparameter_names():
            algorithms = self.cs.get_hyperparameter("algorithm").choices
            self.search_space.append(algorithms)

        self.optimizer_result = {}

    def black_box_function(self, parameters):
        """
        :param parameters: list of parameters. For the k-means algorithm the
        list will only contain the "k" parameter
        :return: Result of the black box function that should be minimized
        """
        k = parameters[0]
        print("k: {}".format(k))
        if len(self.search_space) > 1:
            algorithm = parameters[1]
        else:
            algorithm = ClusteringAlgorithms.KMEANS_ALGORITHM

        clustering_result = ClusteringAlgorithms.run_algorithm(algorithm, self.dataset, k=int(k))
        labels = clustering_result.labels
        self.algo_runtimes.append(clustering_result.execution_time)
        start_metric = time.time()
        score = self.metric.score_metric(self.dataset, labels=labels, true_labels=self.true_labels)
        self.metric_runtimes.append(time.time() - start_metric)
        self.optimizer_runtimes.append(time.time() - self.start_time)
        return score

    @abstractmethod
    def optimize_specific(self):
        """
        Method that runs the optimization process.
        This method differs from the different optimization methods.
        :return: the best parameter configuration that was found by the optimizer.
        """
        pass

    def get_best_configuration(self):
        k = self.optimizer_result[ResultsFields.best_config_history][-1]
        algo = self.optimizer_result[ResultsFields.best_algorithm_history][-1]
        return {"k": k, "algorithm": algo}

    def get_config_history(self):
        result = []
        for k, algo, metric_value in zip(self.optimizer_result[
                                             ResultsFields.best_config_history, ResultsFields.best_algorithm_history, ResultsFields.best_score_history]):
            result.append({"k": k, "algorithm": algo, "metric value": metric_value})
        return result

    def optimize(self):
        result = self.optimize_specific()
        self.optimizer_result = parse_result(result, self)
        return self.optimizer_result

    @staticmethod
    @abstractmethod
    def get_name():
        pass

    @classmethod
    def get_abbrev(cls):
        return OPT_ABBREVS[cls.get_name()]

    def get_configspace(self):
        if self.cs:
            return self.cs
        cs = CS.ConfigurationSpace()
        # Only have 1 parameter so take first parameter of search space list
        # search_space = self.search_space[0]

        algorithm_hyperparameter = CSH.CategoricalHyperparameter("algorithm", choices=ClusteringAlgorithms.algorithms)
        cs.add_hyperparameter(algorithm_hyperparameter)
        k_hyperparameter = CSH.UniformIntegerHyperparameter("k", lower=2,
                                                            upper=200)
        cs.add_hyperparameter(k_hyperparameter)
        self.cs = cs
        return self.cs


class BayesOptimizer(AbstractOptimizer):
    """
        Bayesian Optimizer for Hyperparameter Tuning.
        Based on the skopt library https://scikit-optimize.github.io/notebooks/bayesian-optimization.html
        Not used at the moment since we use the SMAC optimizer for Bayes.
    """

    @staticmethod
    def get_name():
        return "Bayes"

    def __init__(self, dataset, metric, search_space=None, true_labels=None, warmstart_configs=None, file_name=None,
                 rep=0):
        super().__init__(dataset, metric, search_space, true_labels, warmstart_configs, file_name, rep=rep)

    def optimize(self):
        self.start_time = time.time()
        if self.warmstart_configs:
            n_random_starts = len(self.warmstart_configs) != self.n_loops

        else:
            n_random_starts = 1
        print("warmstarts: {}".format(self.warmstart_configs))
        result = gp_minimize(func=self.black_box_function, dimensions=self.search_space, n_calls=self.n_loops,
                             n_random_starts=n_random_starts, x0=self.warmstart_configs)
        parsed_result = parse_result(result, self)
        return parsed_result


class SMACOptimizer(AbstractOptimizer):
    """
        State-of-the-Art Bayesian Optimizer.
         Can be configured to use Random Forests (TPE) or Gaussian Processes.
        However, Gaussian Processes are much slower and only work for low-dimensional parameter spaces.
        Due to this, the default implementation uses Random Forests.
    """

    def smac_function(self, config):
        k = config["k"]
        algorithm = ClusteringAlgorithms.KMEANS_ALGORITHM
        if "algorithm" in config:
            algorithm = config["algorithm"]
        print(Helper.print_timestamp("k: {}".format(k)))
        print(Helper.print_timestamp("algorithm: {}".format(algorithm)))

        clustering_result = run_algorithm(algorithm, self.dataset, k=int(k))
        algo_runtime = clustering_result.execution_time

        labels = clustering_result.labels

        metric_start = time.time()
        score = self.metric.score_metric(self.dataset, labels=labels, true_labels=self.true_labels)
        metric_runtime = time.time() - metric_start

        optimizer_time = time.time() - self.start_time
        metric_algo_runtime = metric_runtime + algo_runtime
        return score, {"opt_time": optimizer_time, "metric_algo_time": metric_algo_runtime}

    def optimize_specific(self):
        self.start_time = time.time()
        # Scenario object
        scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                             "runcount-limit": self.n_loops,  # max. number of function evaluations;
                             "cs": self.get_configspace(),  # configuration space
                             "deterministic": "true"
                             })

        if self.warmstart_configs:
            # if "algorithm" in self.cs.get_hyperparameter_names():
            #    initial_configs = [{'k': c[0], 'algorithm': c[1]} for c in self.warmstart_configs]
            # else:
            #    initial_configs = [{'k': c[0]} for c in self.warmstart_configs]
            initial_configs = [Configuration(configuration_space=self.cs, values=initial_config) for initial_config in
                               self.warmstart_configs]

            self.smac = self.smac_algo(scenario=scenario,
                                       # rng=np.random.RandomState(42),
                                       tae_runner=self.smac_function,
                                       initial_configurations=initial_configs,
                                       initial_design=None)
        else:
            self.smac = self.smac_algo(scenario=scenario,
                                       # rng=np.random.RandomState(42),
                                       tae_runner=self.smac_function)

        #        self.smac.solver.intensifier.tae_runner.use_pynisher = False
        self.smac.optimize()
        return self.smac

    def __init__(self, dataset, metric=None, search_space=None, true_labels=None, warmstart_configs=None, n_loops=60,
                 file_name=None, rep=0,
                 smac=SMAC4HPO, cs=None):
        super().__init__(dataset, metric, search_space, true_labels, warmstart_configs, file_name, cs=cs, rep=rep,
                         n_loops=n_loops)
        self.smac_algo = smac
        self.metric_runtimes = []
        self.algo_runtimes = []
        self.total_times = []

    @staticmethod
    def get_name():
        return "SMAC"

    def update_wallclock_time(self):
        self.optimizer_runtimes.append(self.smac.stats.get_used_wallclock_time())


class RandomOptimizer(AbstractOptimizer):
    """
        Random Optimizer. Basically evaluates random samples and chooses at the end the best result
        based on the given metric.
        Also based on skopt library https://scikit-optimize.github.io/#skopt.dummy_minimize
    """

    @staticmethod
    def get_name():
        return "Random"

    def __init__(self, dataset, metric=None, search_space=None, true_labels=None, warmstart_configs=None,
                 file_name=None, rep=0, cs=None, n_loops=60):
        super().__init__(dataset, metric, search_space, true_labels, warmstart_configs, file_name, rep=rep, cs=cs,
                         n_loops=n_loops)

    def optimize_specific(self):
        if self.warmstart_configs and len(self.search_space) == 1:
            self.warmstarts = [[c["k"]] for c in self.warmstart_configs]
        elif self.warmstart_configs and len(self.search_space) > 1:
            self.warmstarts = [(c["k"], c['algorithm']) for c in self.warmstart_configs]

        self.start_time = time.time()
        if len(search_space) > 1:
            self.search_space = [Integer(self.search_space[0][0], self.search_space[0][1]),
                                 Categorical(self.search_space[1])]
        else:
            self.search_space = [Integer(self.search_space[0][0], self.search_space[0][1])]

        return dummy_minimize(func=self.black_box_function, dimensions=self.search_space, n_calls=self.n_loops,
                              x0=self.warmstarts)


###### Part from the second automl competition for warmstart bohb ######
class PortfolioBOHB(BOHB_generator):
    """ subclasses the config_generator BOHB"""

    def __init__(self, initial_configs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if initial_configs is None:
            # dummy initial portfolio
            self.initial_configs = [self.configspace.sample_configuration().get_dictionary() for i in range(5)]
        else:
            self.initial_configs = initial_configs

    def get_config(self, budget):

        # return a portfolio member first
        if len(self.initial_configs) > 0 and True:
            c = self.initial_configs.pop()
            return (c, {'portfolio_member': True})

        return (super().get_config(budget))

    def new_result(self, job):
        # notify ensemble script or something
        super().new_result(job)


class PortfolioHyperband(RandomSampling):
    """ subclasses the config_generator RandomSampling"""

    def __init__(self, initial_configs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if initial_configs is None:
            # dummy initial portfolio
            self.initial_configs = [self.configspace.sample_configuration().get_dictionary() for i in range(5)]
        else:
            self.initial_configs = initial_configs

    def get_config(self, budget):

        # return a portfolio member first
        if len(self.initial_configs) > 0 and True:
            c = self.initial_configs.pop()
            return (c, {'portfolio_member': True})

        return (super().get_config(budget))

    def new_result(self, job):
        # notify ensemble script or something
        super().new_result(job)


class SuccessivePanicking(BaseIteration):

    def _advance_to_next_stage(self, config_ids, losses):
        """
            SuccessiveHalving simply continues the best based on the current loss.
        """

        if len(config_ids) == 0:
            for i in range(self.stage, len(self.num_configs)):
                self.num_configs[i] = 0

        ranks = np.argsort(np.argsort(losses))
        return (ranks < self.num_configs[self.stage])


class SideShowBOHB(Master):
    def __init__(self, initial_configs=None, configspace=None,
                 eta=3, min_budget=0.01, max_budget=1,
                 min_points_in_model=None, top_n_percent=15,
                 num_samples=64, random_fraction=0.5, bandwidth_factor=3,
                 SH_only=False,
                 run_id=None,
                 bohb=True,
                 **kwargs):
        # MF I changed the parameters a bit to be more aggressive after the
        # portfolio evaluation, but also to still do some random search.

        if bohb:
            cg = PortfolioBOHB(
                initial_configs=initial_configs,
                configspace=configspace,
                min_points_in_model=min_points_in_model,
                top_n_percent=top_n_percent,
                num_samples=num_samples,
                random_fraction=random_fraction,
                bandwidth_factor=bandwidth_factor,
            )
        else:
            cg = PortfolioHyperband(
                initial_configs=initial_configs,
                configspace=configspace,
            )

        super().__init__(config_generator=cg, run_id=run_id, **kwargs)

        # Hyperband related stuff
        self.eta = eta
        self.min_budget = min_budget
        self.max_budget = max_budget

        self.SH_only = SH_only

        # precompute some HB stuff
        self.max_SH_iter = -int(
            np.log(min_budget / max_budget) / np.log(eta)) + 1
        self.budgets = max_budget * np.power(eta,
                                             -np.linspace(self.max_SH_iter - 1,
                                                          0, self.max_SH_iter))
        self.logger.info('Using budgets %s.', self.budgets)

        self.config.update({
            'eta': eta,
            'min_budget': min_budget,
            'max_budget': max_budget,
            'budgets': self.budgets,
            'max_SH_iter': self.max_SH_iter,
            'min_points_in_model': min_points_in_model,
            'top_n_percent': top_n_percent,
            'num_samples': num_samples,
            'random_fraction': random_fraction,
            'bandwidth_factor': bandwidth_factor
        })

    def get_next_iteration(self, iteration, iteration_kwargs={}):
        """
            BO-HB uses (just like Hyperband) SuccessiveHalving for each
            iteration. See Li et al. (2016) for reference.

            Parameters:
            -----------
                iteration: int
                    the index of the iteration to be instantiated
            Returns:
            --------
                SuccessiveHalving: the SuccessiveHalving iteration with the
                    corresponding number of configurations
        """

        # number of 'SH rungs'
        if self.SH_only:
            s = self.max_SH_iter - 1
        else:
            s = self.max_SH_iter - 1 - (iteration % self.max_SH_iter)
        # number of configurations in that bracket
        n0 = int(np.floor((self.max_SH_iter) / (s + 1)) * self.eta ** s)
        ns = [max(int(n0 * (self.eta ** (-i))), 1) for i in range(s + 1)]

        return SuccessivePanicking(HPB_iter=iteration,
                                   num_configs=ns,
                                   budgets=self.budgets[(-s - 1):],
                                   config_sampler=self.config_generator.get_config,
                                   **iteration_kwargs)


class HyperBandWorker(Worker):
    """
        Worker for the hyperband optimization methods.
        Handles initial setup (setting dataset etc.) and also how to deal with one specific configuration (see compute).
        Examples can be found here https://automl.github.io/HpBandSter/build/html/index.html
    """

    def __init__(self, dataset, metric=None, search_space=None, optimizer=None, true_labels=None, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.metric = metric
        self.search_space = search_space
        self.true_labels = true_labels
        self.optimizer = optimizer

    def compute(self, config, budget, working_directory, *args, **kwargs):
        k = config["k"]
        algorithm = ClusteringAlgorithms.KMEANS_ALGORITHM
        if "algorithm" in config:
            algorithm = config["algorithm"]

        clustering_result = run_algorithm(algorithm, data_without_labels=self.dataset, k=k, max_iterations=int(budget))
        metric_start = time.time()
        res = self.metric.score_metric(self.dataset, clustering_result.labels, self.true_labels)
        metric_runtime = time.time() - metric_start
        return ({
            'loss': float(res),  # this is the a mandatory field to run hyperband
            'info': {"algo": algorithm,
                     "algo_time": clustering_result.execution_time,
                     "metric_time": metric_runtime},  # can be used for any user-defined information - also mandatory
        })


# Enable this to see what the hyperband optimizer is doing
# logging.basicConfig(level=logging.INFO)


class HyperBandOptimizer(AbstractOptimizer):
    """
        Problem of Bayesian optimization is that it takes long to evaluate and to converge.
        It also does not scale well. Hyperband tries to overcome this by first random "sampling" and then run them
        for a specific budget. The most promising approaches are also used in the next iteration,
        so they are running longer. See Li et al. (2016) for reference.
        Also based on HpBandster https://automl.github.io/HpBandSter/build/html/index.html
    """

    min_budget = 1
    max_budget = 9
    host = '127.0.0.1'
    min_n_workers = 20
    is_bohb = False

    @staticmethod
    def get_name():
        return "Hyperband"

    def __init__(self, dataset, metric=None, search_space=None, true_labels=None, warmstart_configs=None,
                 file_name=None, rep=0, cs=None, n_loops=60):
        super().__init__(dataset, metric, search_space, true_labels, warmstart_configs, file_name, rep=rep, cs=cs,
                         n_loops=n_loops)
        self.run_id = str(uuid.uuid4())

        self.workers = []
        for i in range(self.min_n_workers):
            w = HyperBandWorker(run_id=self.run_id, dataset=self.dataset,
                                metric=self.metric,
                                true_labels=self.true_labels, search_space=self.search_space, id=i, optimizer=self)
            self.workers.append(w)
        self.cs = self.get_configspace()
        # Set optimizer to HyperBand, this is the only thing that changes between Hyperband and BOHB
        self.hb_optimizer = SideShowBOHB
        self.ns = hpns.NameServer(run_id=self.run_id, host=self.host, port=None)

    def run_workers(self):
        for w in self.workers:
            w.run(background=True)

    def optimize(self):
        self.ns.start()
        self.run_workers()

        print(self.cs)
        print(self.warmstart_configs)
        """
        if "algorithm" in self.cs.get_hyperparameter_names():
            initial_configs = [{'k': c[0], 'algorithm': c[1]} for c in self.warmstart_configs] \
                if self.warmstart_configs else None
        else:
            initial_configs = [{'k': c[0]} for c in self.warmstart_configs] if self.warmstart_configs else None
        """
        initial_configs = self.warmstart_configs
        optimizer_instance = self.hb_optimizer(configspace=self.cs,
                                               initial_configs=initial_configs,
                                               run_id=self.run_id, nameserver=self.host,
                                               min_budget=self.min_budget, max_budget=self.max_budget,
                                               bohb=self.is_bohb,
                                               eta=3)
        res = optimizer_instance.run(n_iterations=self.n_loops, min_n_workers=self.min_n_workers)

        optimizer_instance.shutdown(shutdown_workers=True)
        self.ns.shutdown()
        return res


class BOHBOptimizer(HyperBandOptimizer):
    """
        Combination of Hyperband and Bayesian Optimization. Hyperband does just take random samples and does not take
        the evaluation of the samples before into account, so this method tries to do this by also using bayesian
        optimization to sample the next point.
        Based on HpBandster https://automl.github.io/HpBandSter/build/html/index.html
        @InProceedings{falkner-icml-18,
            title =        {{BOHB}: Robust and Efficient Hyperparameter Optimization at Scale},
            author =       {Falkner, Stefan and Klein, Aaron and Hutter, Frank},
            booktitle =    {Proceedings of the 35th International Conference on Machine Learning},
            pages =        {1436--1445},
            year =         {2018},
        }
    """

    @staticmethod
    def get_name():
        return "BOHB"

    def __init__(self, dataset, metric=None, search_space=None, true_labels=None, warmstart_configs=None, n_loops=60,
                 file_name=None, rep=0, cs=None):
        super().__init__(dataset, metric, search_space, true_labels, warmstart_configs, file_name, rep=rep, cs=cs,
                         n_loops=n_loops)
        self.hb_optimizer = SideShowBOHB
        # Only difference in the code for both optimizers is the instance of the optimizers
        # So use BOHB for this
        self.is_bohb = True


OPT_ABBREVS = {
    # BayesOptimizer.get_name(): "BO",
    SMACOptimizer.get_name(): "BO",
    RandomOptimizer.get_name(): "RS",
    HyperBandOptimizer.get_name(): "HB",
    BOHBOptimizer.get_name(): "BOHB"}


def get_opt_by_abbrev(opt_abbrev):
    if opt_abbrev == OPT_ABBREVS[SMACOptimizer.get_name()]:
        return SMACOptimizer
    elif opt_abbrev == OPT_ABBREVS[RandomOptimizer.get_name()]:
        return RandomOptimizer
    elif opt_abbrev == OPT_ABBREVS[HyperBandOptimizer.get_name()]:
        return HyperBandOptimizer
    elif opt_abbrev == OPT_ABBREVS[BOHBOptimizer.get_name()]:
        return BOHBOptimizer


def get_best_hyperband_config(result):
    id2conf = result.get_id2config_mapping()
    inc_id = result.get_incumbent_id()
    # let's grab the run on the highest budget
    inc_runs = result.get_runs_by_id(inc_id)
    inc_run = inc_runs[-1]

    # We have access to all information: the config, the loss observed during
    # optimization, and all the additional information
    inc_loss = inc_run.loss
    inc_config = id2conf[inc_id]['config']['k']
    return inc_config, inc_loss


def get_ta_runtimes_smac(out_dir):
    """
    parses the runtimes for the ta (algorithm + metric) from the runhistory.json file in the directory out_dir.
    :param out_dir:
    :return: tuple with first element the list that contains the runtime for the target algorithm in each iteration and
    the second element the list of runtimes for the optimizer since the start for each iteration
    """
    ta_times = []
    opt_times = []
    with open('{}/runhistory.json'.format(out_dir)) as json_file:
        data = json.load(json_file)
        data = data['data']
        for runs in data:
            run = runs[1]

            add_info = run[3]

            ta_time = add_info["metric_algo_time"]
            ta_times.append(ta_time)

            optimizer_time = add_info["opt_time"]
            opt_times.append(optimizer_time)

    return ta_times, opt_times


def get_algo_time_by_confid_and_budget(result, conf_id):
    runs_for_config = result.get_runs_by_id(conf_id)
    algo_time_by_conf_id_and_budget = {}
    for run in runs_for_config:
        algo_metric_time = run.info['algo_time'] + run.info['metric_time']
        algo_time_by_conf_id_and_budget[(run.config_id, run["budget"])] = algo_metric_time

    return algo_time_by_conf_id_and_budget


def get_runtime_for_iteration(conf_per_iteration, result):
    """
    Returns the runtime for each Hyperband iteration. Since multiple configs are executed in parallel we retrieve the
    runtime by taking the time_stamps
    :param conf_per_iteration:
    :param result:
    :return:
    """
    all_runs_for_iteration = []
    for conf in conf_per_iteration:
        runs = result.get_runs_by_id(conf)
        for run in runs:
            all_runs_for_iteration.append(run)

    start_of_optimization = result.get_all_runs()[0].time_stamps['started']
    end_of_iteration = max([run.time_stamps['finished'] for run in all_runs_for_iteration])
    return end_of_iteration - start_of_optimization


def get_overall_runtime(all_runs):
    starts = [run.time_stamps['started'] for run in all_runs]
    ends = [run.time_stamps['finished'] for run in all_runs]
    # Return the diff between the earliest started and the latest finished run
    return max(ends) - min(starts)


def parse_hyperband_result(result, optimizer):
    opt_result_dict = OptResultDict()
    logging.info("Optimizer result is instance of Result, so either hyperband or bohb")

    learning_curve = result.get_learning_curves()
    mapping = result.get_id2config_mapping()

    for conf_id, i in zip(learning_curve, range(len(learning_curve))):
        # one config, but it can contain a list if that config is taken to next round and executed with higher budget
        budgets_for_config = learning_curve[conf_id][0]
        iteration = conf_id[0]

        # retrieve configs for each iteration -> we want to use them for retrieving the time for each iteration
        conf_per_iteration = list(filter(lambda conf: conf[0] == iteration, learning_curve))
        # algo_time_for_iteration = get_runtime_for_iteration(conf_per_iteration, result)

        # Workaround, we have to get the runs and the "info" field for the config
        # However, we cannot satisfy the correct ordering, so we save the algo time by conf_id and budget
        algo_time_by_conf_id_and_budget = get_algo_time_by_confid_and_budget(result, conf_id)
        iteration_runtime = get_runtime_for_iteration(conf_per_iteration, result)

        for budget_loss in budgets_for_config:
            score = budget_loss[1]
            budget = budget_loss[0]
            max_budget = max([x[0] for x in budgets_for_config])
            algo_time = algo_time_by_conf_id_and_budget[(conf_id, budget)]

            k = mapping[conf_id]['config']['k']
            if 'algorithm' in mapping[conf_id]['config']:
                algorithm = mapping[conf_id]['config']['algorithm']
            else:
                algorithm = ClusteringAlgorithms.KMEANS_ALGORITHM

            opt_result_dict[ResultsFields.budget_history].append(budget)
            # add one since it starts to count with 0
            # iterations.append(iteration+1)
            opt_result_dict[ResultsFields.iteration].append(iteration + 1)

            opt_result_dict[ResultsFields.score_history].append(score)
            opt_result_dict[ResultsFields.config_history].append(k)
            opt_result_dict[ResultsFields.algorithm_history].append(algorithm)
            opt_result_dict[ResultsFields.algorithm_metric_time_history].append(algo_time)

            logging.info(
                "config: {}, iteration: {}, budget: {}, loss: {}".format(mapping[conf_id]['config']['k'], iteration,
                                                                         budget_loss[0], budget_loss[1]))
            if max_budget == optimizer.max_budget and score < opt_result_dict.best_score:
                opt_result_dict.upate_best_config(score, k, algorithm)

        # We want to keep track of only the best config for the whole iteration, so we do add the best config
        # len(budgets_for_config) times, which is the number how often one config is evaluated
        opt_result_dict[ResultsFields.best_score_history].extend(
            [opt_result_dict.best_score for n in budgets_for_config])
        opt_result_dict[ResultsFields.best_config_history].extend(
            [opt_result_dict.best_k for n in budgets_for_config])
        opt_result_dict[ResultsFields.total_time_per_iteration].extend([iteration_runtime for n in budgets_for_config])
        opt_result_dict[ResultsFields.best_algorithm_history].extend(
            [opt_result_dict.best_algorithm for n in budgets_for_config])

    return opt_result_dict


def parse_random_opt_result(result, optimizer):
    opt_result_dict = OptResultDict()
    logging.info("result is an OptimizeResult object (so either random or bo)")

    # parameters are list of parameters, but we only have one (k) so we are mapping it to the first
    # parameter
    parameter_values = list(map(lambda x: x[0], result.x_iters))
    if len(result.x_iters[0]) > 1:
        algo_history = list(map(lambda x: x[1], result.x_iters))
    else:
        algo_history = [ClusteringAlgorithms.KMEANS_ALGORITHM for x in result.x_iters]
    opt_result_dict[ResultsFields.config_history] = parameter_values
    opt_result_dict[ResultsFields.score_history] = result.func_vals
    opt_result_dict[ResultsFields.algorithm_history] = algo_history

    for ind, k in enumerate(parameter_values):
        algorithm = algo_history[ind]
        score = result.func_vals[ind]
        if score < opt_result_dict.best_score:
            opt_result_dict.upate_best_config(score, k, algorithm)

        opt_result_dict[ResultsFields.best_config_history].append(opt_result_dict.best_k)
        opt_result_dict[ResultsFields.best_score_history].append(opt_result_dict.best_score)
        opt_result_dict[ResultsFields.best_algorithm_history].append(opt_result_dict.best_algorithm)

    opt_result_dict[ResultsFields.algorithm_metric_time_history] = [algo_time + metric_time
                                                                    for algo_time, metric_time
                                                                    in zip(optimizer.algo_runtimes,
                                                                           optimizer.metric_runtimes)]
    opt_result_dict[ResultsFields.iteration] = [ind + 1 for ind in range(len(parameter_values))]
    opt_result_dict[ResultsFields.total_time_per_iteration] = optimizer.optimizer_runtimes
    return opt_result_dict


def parse_smac_result(result, optimizer):
    opt_result_dict = OptResultDict()
    logging.info("Optimizer result is SMBO (SMAC) result.")

    out_dir = result.output_dir
    # get the target algorithm runtimes for each evaluated configuration
    metric_algo_time_history, total_runtimes_iteration_from_start = get_ta_runtimes_smac(out_dir)
    opt_result_dict.set_field(ResultsFields.algorithm_metric_time_history, metric_algo_time_history)
    opt_result_dict.set_field(ResultsFields.total_time_per_iteration, total_runtimes_iteration_from_start)

    result = result.solver
    history = result.runhistory
    all_configs = history.get_all_configs()

    # save history for each config in each iteration
    for config in all_configs:
        score = history.get_cost(config)
        k = config['k']
        algorithm = config["algorithm"]

        if not algorithm:
            algorithm = ClusteringAlgorithms.KMEANS_ALGORITHM

        if score < opt_result_dict.best_score:
            opt_result_dict.upate_best_config(score, k, algorithm)

        # k_deviation_history.append(true_k - best_config)
        opt_result_dict.add_best_config_and_score()

    opt_result_dict.set_field(ResultsFields.iteration, [iteration for iteration, _ in enumerate(all_configs, 1)])
    opt_result_dict.set_field(ResultsFields.score_history, [history.get_cost(conf) for conf in all_configs])
    opt_result_dict.set_field(ResultsFields.config_history, [config['k'] for config in all_configs])
    opt_result_dict.set_field(ResultsFields.algorithm_history, [config['algorithm']
                                                                if config['algorithm']
                                                                else ClusteringAlgorithms.KMEANS_ALGORITHM
                                                                for config in all_configs])

    return opt_result_dict


def parse_result(result, optimizer):
    opt_result_dict = OptResultDict()

    # check if hyperband or bohb result
    if isinstance(result, Result):
        opt_result_dict = parse_hyperband_result(result, optimizer)

    # we don't have instance of Result so we should have either Random or SMAC optimizer
    elif isinstance(result, OptimizeResult):
        opt_result_dict = parse_random_opt_result(result, optimizer)

    # So should be SMAC
    elif isinstance(result, SMAC4HPO):
        opt_result_dict = parse_smac_result(result, optimizer)

    else:
        # Something weird happened
        logging.warning("Found unknown optimizer result {} !".format(result))

    return opt_result_dict


class ResultsFields:
    algorithm_history = "algorithm"
    iteration = "iteration"
    config_history = "k"
    budget_history = "budget"
    score_history = "score"
    best_config_history = "best config"
    best_score_history = "best score"
    # total optimizer time
    # total_optimizer_time = "total time"
    # total runtime for the optimizer per iteration since the start of the optimization
    total_time_per_iteration = "time per iteration"
    # k_deviation_history = "k_deviation"
    # runtime for algorithm and metric per iteration
    algorithm_metric_time_history = "algorithm + metric time"
    best_algorithm_history = "best algorithm"


class OptResultDict(dict):
    def __init__(self, *args, **kwargs):
        result_fields = ResultsFields()
        fields = dir(result_fields)
        [self.init_field(field) for field in fields if not field.startswith("__")]
        self.best_k = -1
        self.best_score = inf
        self.best_algorithm = ClusteringAlgorithms.KMEANS_ALGORITHM

    def init_field(self, field):
        self[getattr(ResultsFields, field)] = []

    def create_dataframe(self):
        if len(self[ResultsFields.budget_history]) == 0:
            self[ResultsFields.budget_history] = [-1 for n in self[ResultsFields.iteration]]
        return pd.DataFrame(self)

    def upate_best_config(self, score, k, algorithm):
        self.best_score = score
        self.best_k = k
        self.best_algorithm = algorithm

    def add(self, field, value):
        self[field].append(value)

    def set_field(self, field, values):
        self[field] = values

    def add_best_config_and_score(self):
        self[ResultsFields.best_config_history].append(self.best_k)
        self[ResultsFields.best_score_history].append(self.best_score)
        self[ResultsFields.best_algorithm_history].append(self.best_algorithm)

    def overwrite_by_dataframe(self, df):
        for key in df.keys():
            self[key] = df[key].values.tolist()
        return self

    def print(self):
        for k, v in self.items():
            print("values for key {k}: {v}".format(k=k, v=v))
