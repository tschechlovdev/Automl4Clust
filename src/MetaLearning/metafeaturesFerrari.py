import itertools
import typing

import numpy as np
import pandas as pd
import scipy.stats
import scipy.spatial


def _is_integer(val: typing.Union[int, float, np.number]):
    """Check whether a numeric value is an integer.

    >>> _is_integer(1)
    True
    >>> _is_integer(3.1)
    False
    >>> _is_integer(3.)
    True
    >>> _is_integer(np.int8(1))
    True
    >>> _is_integer(np.float_(3.1))
    False
    >>> _is_integer(np.longdouble(3.))
    True
    """
    if isinstance(val, (int, np.integer)):
        return True
    return float(val).is_integer()


def _is_discrete(col: np.ndarray):
    """Determine if this column is discrete.

    This follows the rules from the paper (Ferrari 2015):

    1. If the attribute has real numbers, then the attribute is continuous.
    2. If the number of unique values is less than 30%, then the attribute is discrete.
    3. If otherwise, the attribute is continuous.

    >>> _is_discrete(np.array([1, 2, 3, 4]))
    False
    >>> _is_discrete(np.array([1, 1, 1, 1, 1, 1, 1, 2]))
    True
    >>> _is_discrete(np.array([1, 1, 1, 1, 1, 1, 1, 2.]))
    True
    >>> _is_discrete(np.array([1, 1, 1, 1, 1, 1, 1, 2.1]))
    False
    """
    for value in col:
        if not _is_integer(value):
            return False
    if (np.unique(col).shape[0] / col.shape[0]) < 0.3:
        return True
    return False


class MFEFerrari2015Attributes:
    """Custom pymfe group for the attribute-based meta-features used by [Ferrari 2015]_

    .. [Ferrari 2015] Ferrari, D.G. and De Castro, L.N., 2015.
    Clustering algorithm selection by meta-learning systems: A new distance-based problem characterization
    and ranking combination methods.
    Information Sciences, 301, pp.181-194.
    """

    @classmethod
    def precompute_column_types(cls, X: np.ndarray, **kwargs):
        """
        Determine for each column whether it is discrete or continuous.
        """
        precomp_vals = {}

        if X is not None and not {'discrete_columns', 'continuous_columns'}.issubset(kwargs):
            discrete_cols = np.apply_along_axis(_is_discrete, axis=0, arr=X)
            precomp_vals['discrete_columns'] = np.where(discrete_cols)[0]
            precomp_vals['continuous_columns'] = np.where(np.logical_not(discrete_cols))[0]
        return precomp_vals

    @classmethod
    def ft_ma1(cls, X: np.ndarray):
        """log2 of the number of objects"""
        return np.log2(X.shape[0])

    @classmethod
    def ft_ma2(cls, X: np.ndarray):
        """log2 of the number of attributes"""
        return np.log2(X.shape[1])

    @classmethod
    def ft_ma3(cls, X: np.ndarray, discrete_columns: np.ndarray = None):
        """Percentage of discrete attributes"""
        if discrete_columns is None:
            discrete_columns = np.where(np.apply_along_axis(_is_discrete, axis=0, arr=X))[0]
        return discrete_columns.shape[0] / X.shape[1]

    @classmethod
    def ft_ma4(cls, N: np.ndarray):
        """Percentage of outliers

        From the paper
        --------------

        To extract MA_4, a method based on the boxplot is used.
        The lower and upper limits for each attribute is calculated as follows:
            Lower Limit = Q1 - (1.5 * IQR)
            Upper Limit = Q3 + (1.5 * IQR)
        where Q1 is the first quartile, Q3 is the third quartile, and IQR is the interquartile range (Q3–Q1).
        An object is considered an outlier if at least one of its attributes has a value outside its respective upper or lower limits.

        :param N: Fitted numerical data.

        >>> MFEFerrari2015Attributes.ft_ma4(np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [9, 1, 1, 1]]))
        0.25
        """
        # these are per-column
        quartile_1, quartile_3 = np.percentile(N, (25, 75), axis=0)

        whis = 1.5  # default chosen by the paper
        whis_iqr = whis * (quartile_3 - quartile_1)

        lower_limit = quartile_1 - whis_iqr
        upper_limit = quartile_3 + whis_iqr

        def _object_has_outlier(row: np.ndarray) -> np.ndarray:
            return np.any(np.logical_or(lower_limit > row, upper_limit < row))

        num_outliers = np.sum(np.apply_along_axis(_object_has_outlier, axis=1, arr=N))
        return num_outliers / N.shape[0]

    @classmethod
    def ft_ma5(cls, N: np.ndarray, discrete_columns: np.ndarray = None):
        """Mean entropy of discrete attributes

        >>> MFEFerrari2015Attributes.ft_ma5(np.array([[1, 1, 1, 1, 1, 1, 1, 2], [1, 1, 1, 1, 1, 1, 1, 2]]).T)
        0.7309301386270429
        """
        if discrete_columns is None:
            discrete_columns = np.where(np.apply_along_axis(_is_discrete, axis=0, arr=N))[0]

        if len(discrete_columns) == 0:
            return 0

        def _calc_entropy(col: np.ndarray) -> np.ndarray:
            value_freqs = np.unique(col, return_counts=True)
            return scipy.stats.entropy(value_freqs, base=2)

        return np.mean(np.apply_along_axis(_calc_entropy, axis=0, arr=N[:, discrete_columns]))

    @classmethod
    def ft_ma6(cls, N: np.ndarray, discrete_columns: np.ndarray = None):
        """Mean concentration between discrete attributes

        >>> MFEFerrari2015Attributes.ft_ma6(np.array([[1, 1, 1, 1, 1, 1, 1, 2], [1, 1, 1, 1, 1, 1, 1, 9]]).T)
        1.0000000000000082
        """
        if discrete_columns is None:
            discrete_columns = np.where(np.apply_along_axis(_is_discrete, axis=0, arr=N))[0]

        if len(discrete_columns) < 2:
            return 0

        col_combinations = itertools.combinations(discrete_columns, 2)

        # NOTE: taken from pymfe!
        def _calc_conc(vec_x: np.ndarray, vec_y: np.ndarray, epsilon: float = 1.0e-8) -> float:
            # Assess whether concentration in the paper == Concentration coefficient
            """Concentration coefficient between two arrays ``vec_x`` and ``vec_y``."""
            pij = pd.crosstab(vec_x, vec_y, normalize=True).values + epsilon

            isum = pij.sum(axis=0)
            jsum2 = np.sum(pij.sum(axis=1) ** 2)

            conc = (np.sum(pij ** 2 / isum) - jsum2) / (1.0 - jsum2)

            return conc

        attr_conc = np.array([
            _calc_conc(N[:, ind_attr_a], N[:, ind_attr_b])
            for ind_attr_a, ind_attr_b in col_combinations
        ])
        return np.mean(attr_conc)

    @classmethod
    def ft_ma7(cls, N: np.ndarray, continuous_columns: np.ndarray = None):
        """Mean absolute correlation between continuous attributes

        """
        if continuous_columns is None:
            continuous_columns = np.where(np.apply_along_axis(lambda col: not _is_discrete(col), axis=0, arr=N))[0]

        if len(continuous_columns) < 2:
            return 0

        col_combinations = itertools.combinations(continuous_columns, 2)

        attr_corr = np.array([
            scipy.spatial.distance.correlation(N[:, ind_attr_a], N[:, ind_attr_b])
            for ind_attr_a, ind_attr_b in col_combinations
        ])

        return np.mean(attr_corr)

    @classmethod
    def ft_ma8(cls, N: np.ndarray, continuous_columns: np.ndarray = None):
        """Mean skewness of continuous attributes

        """
        if continuous_columns is None:
            continuous_columns = np.where(np.apply_along_axis(lambda col: not _is_discrete(col), axis=0, arr=N))[0]

        if len(continuous_columns) == 0:
            return 0

        return np.mean(np.apply_along_axis(scipy.stats.skew, axis=0, arr=N[:, continuous_columns]))

    @classmethod
    def ft_ma9(cls, N: np.ndarray, continuous_columns: np.ndarray = None):
        """Mean kurtosis of continuous attributes

        """
        if continuous_columns is None:
            continuous_columns = np.where(np.apply_along_axis(lambda col: not _is_discrete(col), axis=0, arr=N))[0]

        if len(continuous_columns) == 0:
            return 0

        return np.mean(np.apply_along_axis(scipy.stats.kurtosis, axis=0, arr=N[:, continuous_columns]))


def _normalized(a, axis=-1, order=2):
    """

    >>> _normalized(np.array([5, 5]))
    array([[0.70710678, 0.70710678]])
    >>> _normalized(np.array([5, 5]), order=1)
    array([[0.5, 0.5]])
    """
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def _make_distance_vector(X: np.ndarray):
    """
    Build a distance vector of the form

    d = [d_1_2, d_1_3, ..., d_i_j, d_n-2_n-1, , d_n-1,n]

    normalized to [0, 1]

    >>> _make_distance_vector(np.array([[1, 1], [2, 1], [3, 1]]))
    array([[0.25, 0.5 , 0.25]])
    """
    return _normalized(np.array([scipy.spatial.distance.euclidean(u, v) for u, v in itertools.combinations(X, 2)]),
                       order=1)


def _collect_distance_percentages(distances: np.ndarray):
    """
    Calculate percentage of values in the interval [x, x+0.1], respective (x, x+0.1] for x > 0

    >>> _collect_distance_percentages(np.array([0.1, 0.1, 0.1, 0.2]))
    [0.75, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    """
    # Note that we can't use np.histogram here, as their specification is a tiny bit different when it comes
    # to whether a bin is half-open or not.
    # The paper specifies the first bin to be closed, in the case of numpy however, the last one is closed.
    counts = [0] * 10
    if len(distances.shape) > 1:
        distances = distances.reshape(distances.shape[1])

    for d in distances:
        if d <= 0.1:
            counts[0] += 1
        elif d <= 0.2:
            counts[1] += 1
        elif d <= 0.3:
            counts[2] += 1
        elif d <= 0.4:
            counts[3] += 1
        elif d <= 0.5:
            counts[4] += 1
        elif d <= 0.6:
            counts[5] += 1
        elif d <= 0.7:
            counts[6] += 1
        elif d <= 0.8:
            counts[7] += 1
        elif d <= 0.9:
            counts[8] += 1
        elif d <= 1.0:
            counts[9] += 1
    return [float(c) / distances.shape[-1] for c in counts]


def _calculate_zscore_percentages(distances: np.ndarray):
    if len(distances.shape) > 1:
        distances = distances.reshape(distances.shape[1])
    print("distances: {}".format(distances))
    z = scipy.stats.zscore(distances)
    z[z < 0] = z[z < 0] * (-1)
    print("z:  {}".format(z))
    hist = np.histogram(z, bins=[0, 1, 2, 3, np.inf])
    print("hist: {}".format(hist))
    z_score_percentages = [float(c) / len(distances) for c in hist[0]]
    assert sum(z_score_percentages) == 1
    return z_score_percentages


class MFEFerrari2015Distance:
    """Custom pymfe group for the distance-based meta-features used by [Ferrari 2015]_

    .. [Ferrari 2015] Ferrari, D.G. and De Castro, L.N., 2015.
    Clustering algorithm selection by meta-learning systems: A new distance-based problem characterization
    and ranking combination methods.
    Information Sciences, 301, pp.181-194.
    """

    @classmethod
    def precompute_distance_matrix(cls, X: np.ndarray, **kwargs):
        """
        Precompute the distance matrix for all objects / instances.
        """
        precomp_vals = {}

        if X is not None and not {'distances', 'distance_percentages'} not in kwargs:
            distances = _make_distance_vector(X)
            precomp_vals['distances'] = distances
            precomp_vals['distance_percentages'] = _collect_distance_percentages(distances)
        return precomp_vals

    @classmethod
    def ft_md1(cls, X: np.ndarray, distances: np.ndarray = None):
        """Mean of d"""
        if distances is None:
            distances = _make_distance_vector(X)
        return np.mean(distances)

    @classmethod
    def ft_md2(cls, X: np.ndarray, distances: np.ndarray = None):
        """Variance of d"""
        if distances is None:
            distances = _make_distance_vector(X)
        return np.var(distances)

    @classmethod
    def ft_md3(cls, X: np.ndarray, distances: np.ndarray = None):
        """Standard deviation of d"""
        if distances is None:
            distances = _make_distance_vector(X)
        return np.std(distances)

    @classmethod
    def ft_md4(cls, X: np.ndarray, distances: np.ndarray = None):
        """Skewness of d"""
        if distances is None:
            distances = _make_distance_vector(X)
        return scipy.stats.skew(distances)

    @classmethod
    def ft_md5(cls, X: np.ndarray, distances: np.ndarray = None):
        """Kurtosis of d"""
        if distances is None:
            distances = _make_distance_vector(X)
        return scipy.stats.kurtosis(distances)

    @classmethod
    def ft_md6(cls, X: np.ndarray, distances: np.ndarray = None, distance_percentages: typing.List[float] = None):
        """% of values in the interval [0,0.1]"""
        if distances is None:
            distances = _make_distance_vector(X)
        if distance_percentages is None:
            distance_percentages = _collect_distance_percentages(distances)
        return distance_percentages[0]

    @classmethod
    def ft_md7(cls, X: np.ndarray, distances: np.ndarray = None, distance_percentages: typing.List[float] = None):
        """% of values in the interval (0.1,0.2]"""
        if distances is None:
            distances = _make_distance_vector(X)
        if distance_percentages is None:
            distance_percentages = _collect_distance_percentages(distances)
        return distance_percentages[1]

    @classmethod
    def ft_md8(cls, X: np.ndarray, distances: np.ndarray = None, distance_percentages: typing.List[float] = None):
        """% of values in the interval (0.2,0.3]"""
        if distances is None:
            distances = _make_distance_vector(X)
        if distance_percentages is None:
            distance_percentages = _collect_distance_percentages(distances)
        return distance_percentages[2]

    @classmethod
    def ft_md9(cls, X: np.ndarray, distances: np.ndarray = None, distance_percentages: typing.List[float] = None):
        """% of values in the interval (0.3,0.4]"""
        if distances is None:
            distances = _make_distance_vector(X)
        if distance_percentages is None:
            distance_percentages = _collect_distance_percentages(distances)
        return distance_percentages[3]

    @classmethod
    def ft_md10(cls, X: np.ndarray, distances: np.ndarray = None, distance_percentages: typing.List[float] = None):
        """% of values in the interval (0.4,0.5]"""
        if distances is None:
            distances = _make_distance_vector(X)
        if distance_percentages is None:
            distance_percentages = _collect_distance_percentages(distances)
        return distance_percentages[4]

    @classmethod
    def ft_md11(cls, X: np.ndarray, distances: np.ndarray = None, distance_percentages: typing.List[float] = None):
        """% of values in the interval (0.5,0.6]"""
        if distances is None:
            distances = _make_distance_vector(X)
        if distance_percentages is None:
            distance_percentages = _collect_distance_percentages(distances)
        return distance_percentages[5]

    @classmethod
    def ft_md12(cls, X: np.ndarray, distances: np.ndarray = None, distance_percentages: typing.List[float] = None):
        """% of values in the interval (0.6,0.7]"""
        if distances is None:
            distances = _make_distance_vector(X)
        if distance_percentages is None:
            distance_percentages = _collect_distance_percentages(distances)
        return distance_percentages[6]

    @classmethod
    def ft_md13(cls, X: np.ndarray, distances: np.ndarray = None, distance_percentages: typing.List[float] = None):
        """% of values in the interval (0.7,0.8]"""
        if distances is None:
            distances = _make_distance_vector(X)
        if distance_percentages is None:
            distance_percentages = _collect_distance_percentages(distances)
        return distance_percentages[7]

    @classmethod
    def ft_md14(cls, X: np.ndarray, distances: np.ndarray = None, distance_percentages: typing.List[float] = None):
        """% of values in the interval (0.8,0.9]"""
        if distances is None:
            distances = _make_distance_vector(X)
        if distance_percentages is None:
            distance_percentages = _collect_distance_percentages(distances)
        return distance_percentages[8]

    @classmethod
    def ft_md15(cls, X: np.ndarray, distances: np.ndarray = None, distance_percentages: typing.List[float] = None):
        """% of values in the interval (0.9,1.0]"""
        if distances is None:
            distances = _make_distance_vector(X)
        if distance_percentages is None:
            distance_percentages = _collect_distance_percentages(distances)
        return distance_percentages[9]

    @classmethod
    def ft_md16(cls, X: np.ndarray, distances: np.ndarray = None, zscore_percentages: typing.List[float] = None):
        """% of values with absolute Z-score in the interval [0,1)"""
        if distances is None:
            distances = _make_distance_vector(X)
        if zscore_percentages is None:
            zscore_percentages = _calculate_zscore_percentages(distances)
        print("z score percentages: {}".format(zscore_percentages))
        return zscore_percentages[0]

    @classmethod
    def ft_md17(cls, X: np.ndarray, distances: np.ndarray = None, zscore_percentages: typing.List[float] = None):
        """% of values with absolute Z-score in the interval [1,2)"""
        if distances is None:
            distances = _make_distance_vector(X)
        if zscore_percentages is None:
            zscore_percentages = _calculate_zscore_percentages(distances)
        return zscore_percentages[1]

    @classmethod
    def ft_md18(cls, X: np.ndarray, distances: np.ndarray = None, zscore_percentages: typing.List[float] = None):
        """% of values with absolute Z-score in the interval [2,3)"""
        if distances is None:
            distances = _make_distance_vector(X)
        if zscore_percentages is None:
            zscore_percentages = _calculate_zscore_percentages(distances)
        return zscore_percentages[2]

    @classmethod
    def ft_md19(cls, X: np.ndarray, distances: np.ndarray = None, zscore_percentages: typing.List[float] = None):
        """% of values with absolute Z-score in the interval [3,1)"""
        if distances is None:
            distances = _make_distance_vector(X)
        if zscore_percentages is None:
            zscore_percentages = _calculate_zscore_percentages(distances)
        return zscore_percentages[3]


# see https://github.com/ealcobaca/pymfe/blob/master/pymfe/_dev.py
from pymfe import _internal

_internal.VALID_GROUPS = (*_internal.VALID_GROUPS, 'ferrari2015attr', 'ferrari2015dist')
_internal.VALID_MFECLASSES = (*_internal.VALID_MFECLASSES, MFEFerrari2015Attributes, MFEFerrari2015Distance)

if __name__ == "__main__":
    import doctest

    doctest.testmod()
    from sklearn.datasets import load_iris
    from sklearn.datasets import make_blobs
    from pymfe.mfe import MFE

    X, y = make_blobs(500, 2)

    print(X.shape)
    mfe = MFE(groups="ferrari2015dist")
    mfe.fit(X[:, 0], y)
    result = mfe.extract()
    print(result)
