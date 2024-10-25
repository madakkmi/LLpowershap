'''
Original authors: Jarne Verhaeghe
Modified by: Iqbal Madakkatel
Date: 15/Dec/2023

'''


import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestPower

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning





import copy


def p_values_arg_coef(coefficients, arg):
    return stats.percentileofscore(coefficients, arg)


def p_value_two_samples(case_sample, control_sample):

    _, p_value = stats.mannwhitneyu(case_sample, control_sample, alternative='greater')   
    return p_value

def cohens_d(delta, pooled_standard_deviation):
    
    effect_size = delta / pooled_standard_deviation
    return effect_size


def glasss_delta(delta, sc):
    
    effect_size = delta / sc
    return effect_size

def powerSHAP_statistical_analysis(
    shaps_df: pd.DataFrame, power_alpha: float, power_req_iterations: float, include_all: bool, method: str
):
    p_values = []
    effect_size = []
    power_list = []
    required_iterations = []
    n_samples = len(shaps_df["random_uniform_feature"].values)

    if method == 'powershap':
        random_feature = "random_uniform_feature"
    else:

        random_feature = 'random_ref_feature'
        shaps_df[random_feature] = shaps_df[["random_uniform_feature", "random_normal_feature", "random_expo_feature", 
        "random_cauchy_feature", "random_logistic_feature"]].max(axis=1)

    mean_random_feature = shaps_df[random_feature].mean()
    m = len(shaps_df.columns)
    for i in range(m):

        if method == 'powershap':
            p_value = p_values_arg_coef(np.array(shaps_df.values[:, i]), mean_random_feature) / 100
        else:     
            if i >= (m - 6):
                p_value = 1.0   
            else:
                p_value = p_value_two_samples(shaps_df.values[:,i], shaps_df[random_feature].values) 

        p_values.append(p_value)

        if include_all or p_value < power_alpha:

            pooled_standard_deviation = np.sqrt(
                (
                    (shaps_df.std().values[i] ** 2)
                    + (shaps_df[random_feature].values.std() ** 2)
                )
                / (2)
            )   
            
            if method == 'llpowershap':
                delta = mean_random_feature - shaps_df.mean().values[i]
                _, levene_pval = stats.levene(shaps_df[random_feature].values, shaps_df.values[:,i])
                if levene_pval < power_alpha:
                    eff = glasss_delta(delta, shaps_df.std().values[i]) # using feature's std
                else:
                    eff = cohens_d(delta, pooled_standard_deviation) 
                effect_size.append(eff)

            else:
             
                effect_size.append(
                    (mean_random_feature - shaps_df.mean().values[i]) / pooled_standard_deviation
                )
     
            power_list.append(
                TTestPower().power(
                    effect_size=effect_size[-1],
                    nobs=n_samples,
                    alpha=power_alpha,
                    df=None,
                    alternative="smaller",
                )
            )
            if shaps_df.columns[i] == random_feature:
                required_iterations.append(0)
            else:
                try:
                    with warnings.catch_warnings():

                        result = TTestPower().solve_power(
                                effect_size=effect_size[-1],
                                nobs=None,
                                alpha=power_alpha,
                                power=power_req_iterations,
                                alternative="smaller",
                            )

                        if isinstance(result, (list, tuple, np.ndarray)):
                            required_iterations.append(result[0])
                        else:
                            required_iterations.append(result)

                except ConvergenceWarning:
                    required_iterations.append(0)

        else:
            required_iterations.append(0)
            effect_size.append(0)
            power_list.append(0)


    processed_shaps_df = pd.DataFrame(
        data=np.hstack(
            [
                np.reshape(shaps_df.mean().values, (-1, 1)),
                np.reshape(np.array(p_values), (len(p_values), 1)),
                np.reshape(np.array(effect_size), (len(effect_size), 1)),
                np.reshape(np.array(power_list), (len(power_list), 1)),
                np.reshape(np.array(required_iterations), (len(required_iterations), 1)),
                np.reshape(shaps_df.std().values, (-1, 1)),
            ]
        ),
        columns=[
            "impact",
            "p_value",
            "effect_size",
            "power_" + str(power_alpha) + "_alpha",
            str(power_req_iterations) + "_power_its_req",
            "impact_std",
        ],
        index=shaps_df.mean().index,
    )

    if method == 'powershap':
        processed_shaps_df.drop(columns=['impact_std'], inplace=True)

        processed_shaps_df = processed_shaps_df.reindex(
            processed_shaps_df.impact.abs().sort_values(ascending=False).index
        )
    else:

        processed_shaps_df = processed_shaps_df.reindex(
            processed_shaps_df.impact.sort_values(ascending=False).index
        )        

    return processed_shaps_df
