# mks_trend.py
#
# Mann-Kendall trend test with Theil-Sen / seasonal Sen's slope, used for
# the trend analysis in Sect. 6 of the manuscript (Fig. 7). Transcribed
# unmodified from the methodology code (W.J. Zaadnoordijk, TNO-GDN) behind
# the Dutch Groundwater head viewer (TNO-GDN, 2024) cited in the paper.
#
# functions:
#   check_period        - period-coverage / resampling helper (not called by
#                          compute_trends.py; kept for methodological fidelity
#                          with the original code)
#   kruskal_wallis       - seasonality check (called by mannkendallsen)
#   mannkendallsen        - the trend test itself
import numpy as np
import pandas as pd
from collections import namedtuple
from scipy.stats import kruskal
from pandas import infer_freq
import pymannkendall as mk


def check_period(dates_seq,
                 measurements_seq, trend_period,
                 start_date, end_date):
    # checks requirements for series:
    # - cover at least 80% of the trend period
    # - have at least 20 measurements in the period
    # and resample:
    # - to monthly values when there is data for at least 70 % of the months
    # - otherwise resample quarterly.

    # select the data in the trend period
    selection_list = [date for date in dates_seq if start_date <= date <= end_date]
    if not selection_list:
        return None, "no measurements within period"
    first_date = selection_list[0]
    last_date = selection_list[-1]
    year_difference = last_date.year - first_date.year
    if last_date.month < first_date.month or  \
        (last_date.month == first_date.month  \
        and last_date.day < first_date.day):
        year_difference -= 1
    # check whether measurements cover 80% of the trend period
    if year_difference >= trend_period*0.8:
        # at least 80% of trend period is covered
        # create dictionary to store data
        selected_measurements = [measurement for date, measurement in zip(dates_seq, measurements_seq) if first_date <= date <= last_date]
        dict_period = { "Dates": selection_list, "Measurements": selected_measurements}
        # check whether there are at least 20 measurements within period
        valid_measurements = [m for m in dict_period["Measurements"] if not pd.isnull(m) and m]
        if len(valid_measurements) < 20:
            return None, "less than 20 measurements within trend period"
        # resample to monthly values, check percentage of missing values
        df = pd.DataFrame({"Dates": dict_period["Dates"], "Measurements": dict_period["Measurements"]})
        df.set_index("Dates", inplace=True)
        monthly_resampled = df["Measurements"].resample("ME").median()
        missing_values_percentage = monthly_resampled.isnull().sum() / len(monthly_resampled)
        if missing_values_percentage > 0.3:
            # if resampling by month misses too many values, try to resample by quarter of a year
            print("monthly resampling: more than 30% missing values")
            quarterly_resampled = df["Measurements"].resample("Q").median()
            missing_values_Q = quarterly_resampled.isnull().sum() / len(quarterly_resampled)
            if missing_values_Q > 0.3:
                print("quarterly resampling: more than 30% missing values")
                return None, ">30% missing in monthly and quarterly resampling"
            else:
                result = quarterly_resampled
                nper_result = 4
        else:
            # monthly resampling: less than 30% of monthly values are missing
            result = monthly_resampled
            nper_result = 12
    else:
        return None, "less than 80% of trend period is covered"
    return result, nper_result


def kruskal_wallis(ts, nper):
    # check the seasonality
    freq = infer_freq(ts.index)
    nper = 4 if freq == "3ME" else 12
    _ = mk.seasonal_sens_slope(ts.to_numpy())
    seas_slope = _.slope
    seas_slope = seas_slope / nper

    t = seas_slope * ts.reset_index().index
    ts_slope = t + (ts - t).median()
    ts_corr = ts - ts_slope
    freq = infer_freq(ts.index)
    if freq != "3ME":
        ts_corr = ts_corr.resample("3ME").median()
    qrt = ts_corr.index.quarter

    # filter out NaN values from each group
    group1 = ts_corr[qrt == 1].dropna().values
    group2 = ts_corr[qrt == 2].dropna().values
    group3 = ts_corr[qrt == 3].dropna().values
    group4 = ts_corr[qrt == 4].dropna().values

    if len(group1) == 0 or len(group2) == 0 or      \
        len(group3) == 0 or len(group4) == 0:
        print("WARNING group with length 0 in Kruskal-Wallis")
        print("qrt", qrt)
        print("ts_corr", ts_corr)
        pval = 1
    else:
        # calculate Kruskal-Wallis for the filtered data
        _, pval = kruskal(group1, group2, group3, group4)

    return pval


def mannkendallsen(resampled, nper, alpha=0.05):
    # function to determine significance of trend using variant of
    #   Mann-Kendall test depending on seasonality in sequence of
    #   equidistant values
    #   and calculate Sen's slope with upper and lower bound
    #   of confidence interval
    # input:
    #   resampled : equidistant series
    #   nper      : number of values per year in resampled
    # returns:
    #   significance : logical indicating whether trend is significant
    #   slope        : slope with difference of value for interval
    #                  between two subseqent values
    #   intercept    : value of trend line at first value of resampled
    #   lower_bound  : lower bound of confidence interval of slope
    #   upper_bound  : upper bound of confidence interval of slope
    # NOTE: the unit of the horizontal coordinate of the slope
    #       is the distance between two values in the series
    res = namedtuple("MannKendall_SensSlope",
        ["significance", "slope", "lower_bound", "upper_bound",
         "intercept", "tau", "p", "z", "s", "var_s"])
    if nper > 1:
        # check seasonality with kruskal wallis
        pval_seasonality = kruskal_wallis(resampled, nper)
    else:
        pval_seasonality = 1

    # calculate slope with the relevant test
    if pval_seasonality > alpha:
        # no seasonality
        if nper == 12:
            # assume autocorrelation
            test = mk.hamed_rao_modification_test(resampled,
               lag=3, alpha=alpha)
            test2 = mk.yue_wang_modification_test(resampled,
               lag=1, alpha=alpha)
            slope_ci = test.slope_ci * nper
        else:
            # assume no autocorrelation
            test = mk.original_test(resampled, alpha=alpha)
            slope_ci = test.slope_ci * nper
        slope = test.slope * nper
    else:
        # seasonality
        if nper == 12:
            # assume autocorrelation
            test = mk.correlated_seasonal_test(resampled,
                period=nper, alpha=alpha)
        else:
            # assume no autocorrelation
            test = mk.seasonal_test(resampled,
                period=nper, alpha=alpha)
        slope_ci = test.slope_ci
        slope = test.slope
    # attributs of test:
    #    trend: tells the trend (increasing, decreasing or no trend)
    #    h: True (if trend is present) or False (if trend is absent)
    #    p: p-value of the significance test
    #    z: normalized test statistics
    #    Tau: Kendall Tau
    #    s: Mann-Kendal's score
    #    var_s: Variance S
    #    slope: Theil-Sen estimator/slope
    #    intercept: intercept of Kendall-Theil Robust Line
    return res(test.p < alpha,
            slope,
            slope_ci[0],
            slope_ci[1],
            test.intercept,
            test.Tau,
            test.p,
            test.z,
            test.s,
            test.var_s)
    # this return value uses for the significance the only the p-value
    # and not the logical h (indicating the presence of a trend)
