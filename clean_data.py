# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_ext_and_nan(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_ctg = {feature: (CTG_features[feature].apply(pd.to_numeric, args=('coerce',))).dropna() for feature in
             CTG_features if feature != extra_feature}

    # --------------------------------------------------------------------------
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    c_cdf = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_ctg = CTG_features.apply(pd.to_numeric, errors='coerce')
    c_ctg = c_ctg.drop(extra_feature, axis=1)
    for feature in c_ctg.columns:
        if c_ctg[feature].isnull().values.any():
            val = np.asarray(c_ctg[feature])
            val = val[~np.isnan(val)]
            idx = c_ctg[c_ctg[feature].isnull()].index.to_numpy()
            c_ctg[feature][idx] = np.random.choice(val, len(idx))
            c_cdf[feature] = np.asarray(c_ctg[feature])
        else:
            c_cdf[feature] = np.asarray(c_ctg[feature])
    # dictionary test
    while True:
        try:
            type(c_cdf) == dict
            break
        except ValueError:
            print("c_cdf is not a dict")
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_cdf)


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    d_summary_temp = c_feat.describe()
    d_summary = d_summary_temp[3:8].to_dict()
    # -------------------------------------------------------------------------
    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_cdf
    :param d_summary: Output of sum_stat
    :return: Dataframe of the dictionary c_no_outlier containing the feature with the outliers removed
    """
    c_no_outlier = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    for feature in c_feat.columns:
        q1 = d_summary[feature]['25%']
        q3 = d_summary[feature]['75%']
        iqr = q3 - q1  # Interquartile range
        fence_low = q1 - 1.5 * iqr
        fence_high = q3 + 1.5 * iqr
        # finding indices of outliers in feature column
        idx = c_feat[(c_feat[feature] > fence_low) & (c_feat[feature] < fence_high)].index.to_numpy
        if np.any(idx):
            c_no_outlier[feature] = np.asarray(c_feat[feature], dtype=object)
            continue
        c_feat[feature][idx] = np.nan
        c_no_outlier[feature] = np.asarray(c_feat[feature], dtype=object)
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_no_outlier)


def phys_prior(c_cdf, feature, thresh):
    """

    :param c_cdf: Output of nan2num_cdf
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    raise NotImplementedError()
    # -------------------------------------------------------------------------
    return filt_feature


def norm_standard(CTG_features, selected_feat=('LB', 'ASTV'), mode='none', flag=False):
    """

    :param CTG_features: Pandas series of CTG features
    :param selected_feat: A two elements tuple of strings of the features for comparison
    :param mode: A string determining the mode according to the notebook
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """
    x, y = selected_feat
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    nsd_res = {}
    # Default
    if mode is None:
        nsd_res = {x: CTG_features[x], y: CTG_features[y]}

    # Feature Scaling
    if mode == 'standard':
        # Standardization
        Z_x = (CTG_features[x] - np.mean(CTG_features[x])) / np.std(CTG_features[x])
        Z_y = (CTG_features[y] - np.mean(CTG_features[y])) / np.std(CTG_features[y])
        nsd_res = {x: Z_x, y: Z_y}
        if flag:
            fig1, ax1 = plt.subplots()
            fig1.suptitle('Standarized data for ' + x)
            ax1.hist(Z_x)
            ax1.set(xlabel=x, ylabel='counts')
            fig2, ax2 = plt.subplots()
            fig2.suptitle('Standarized data for ' + y)
            ax2.hist(Z_y)
            ax2.set(xlabel=y, ylabel='counts')
    if mode == 'MinMax':
        # Normalization
        x_norm = (CTG_features[x] - np.min(CTG_features[x])) / (np.max(CTG_features[x]) - np.min(CTG_features[x]))
        y_norm = (CTG_features[y] - np.min(CTG_features[y])) / (np.max(CTG_features[y]) - np.min(CTG_features[y]))
        nsd_res = {x: x_norm, y: y_norm}
        if flag:
            fig1, ax1 = plt.subplots()
            fig1.suptitle('Normalization data for ' + x)
            ax1.hist(x_norm)
            ax1.set(xlabel=x, ylabel='counts')
            fig2, ax2 = plt.subplots()
            fig2.suptitle('Normalization data for ' + y)
            ax2.hist(y_norm)
            ax2.set(xlabel=y, ylabel='counts')
    if mode == 'mean':
        # Mean normalization
        x_norm = (CTG_features[x] - np.mean(CTG_features[x])) / (np.max(CTG_features[x]) - np.min(CTG_features[x]))
        y_norm = (CTG_features[y] - np.mean(CTG_features[y])) / (np.max(CTG_features[y]) - np.min(CTG_features[y]))
        nsd_res = {x: x_norm, y: y_norm}
        if flag:
            fig1, ax1 = plt.subplots()
            fig1.suptitle('mean Normalization data for ' + x)
            ax1.hist(x_norm)
            ax1.set(xlabel=x, ylabel='counts')
            fig2, ax2 = plt.subplots()
            fig2.suptitle(' mean Normalization data for ' + y)
            ax2.hist(y_norm)
            ax2.set(xlabel=y, ylabel='counts')
    # -------------------------------------------------------------------------
    return pd.DataFrame(nsd_res)
