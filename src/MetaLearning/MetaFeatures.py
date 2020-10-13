"""
Holds value of all metafeatures that are not used for supervised learning! (so where class labels are not needed)
commented metafeatures need class labels.

Also the metafeatures with 'cat' are outcommented since we only have numerical attributes.
"""
META_FEATURE_LIST = [
    'attr_conc',
    'attr_ent',
    'attr_to_inst',
    # 'can_cor',
    # 'cat_to_num',
    # 'class_conc',
    # 'class_ent',
    'cor',
    'cov',
    'eigenvalues',
    # 'eq_num_attr',
    # 'freq_class',
    'g_mean',
    # 'gravity',
    'h_mean',
    'inst_to_attr',
    'iq_range',
    # 'joint_ent',
    'kurtosis',
    'mad',
    'max',
    'mean',
    'median',
    'min',
    # 'mut_inf',
    'nr_attr',
    # 'nr_cat',
    # 'nr_class',
    'nr_cor_attr',
    # 'nr_disc',
    'nr_inst',
    'nr_norm',
    'nr_num',
    'nr_outliers',
    # 'ns_ratio',
    # 'num_to_cat',
    'range',
    'sd',
    # 'sd_ratio',
    'skewness',
    'sparsity',
    't_mean',
    'var']
# 'w_lambda']


SUMMARY_LIST = ["median", "min", "max", "mean"]
