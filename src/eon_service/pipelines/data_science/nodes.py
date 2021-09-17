import os
import sys

from typing import Any, Dict, Tuple, List

import pandas as pd
import numpy as np
import time
import logging

from data_services.sklearn_wrapper import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as sklearn_Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

import mlflow
from mlflow import sklearn as mlf_sklearn
from data_services.mlflow_utils import add_environment_specs_to_conda_file
from data_services.mlflow_utils import init_mlflow_run

from data_services.FilterPipeline import FilterPipeline
from data_services.OrderFeatures import OrderFeatures
from data_services.format_missing_data import FormatMissingData
from data_services.passthrough_estimator import PassthroughEstimator
from data_services.passthrough_residual_estimator import (
    PassthroughResidualEstimator
)
from data_services.timedelta_total_seconds_transformer import (
    AddDurationToTimestampPipeline
)
from data_services.filter_pipeline_utils import (
    add_rules_to_filter_pipeline
)
from data_services.drop_inputs_transformer import DropInputsTransfomer
from data_services.data_inspector import DataInspector
from data_services.data_inspector import append_data_inspector
from data_services.evaluation_utils import *
from data_services.error_metrics import METRIC_NAME_TO_FUNCTION_DICT

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300


def train_model(
    data: pd.DataFrame,
    sklearn_pipeline_impute_steps: List[Tuple],
    sklearn_pipeline_encode_steps: List[Tuple],
    sklearn_pipeline_transform_steps: List[Tuple],
    model_name: str,
    model_params: Dict[str, Any],
    inputs_params: Dict[str, Any],
    inputs_params_train_only: Dict[str, Any],
    target_params: Dict[str, Any],
    globals: Dict[str, Any],
    mlflow_params: Dict[str, Any],
    mlflow_experiment_id: int,
    pipeline_inspect_data_verbosity: int = 0,
    data_inspector_verbosity: int = 0,
) -> Tuple:
    """
    Trains a model

    Note on pipeline_inspect_data_verbosity approach
    0: no data inspectors
    1: only right before going into model
    2: same as 1, plus inspect raw input right at start
    3: after all per-input impute, after all per-input encode,
    and after all per-input transform steps
    4: between all steps
    """

    sklearn_Pipeline_steps = []

    # Always include an OrderFeatures, just to be safe
    order_features = OrderFeatures()
    sklearn_Pipeline_steps.extend(
        [
            ("order_features", order_features),
         ],
        )

    # Initial data inspector of raw input data
    if pipeline_inspect_data_verbosity > 1:
        sklearn_Pipeline_steps = append_data_inspector(
            sklearn_Pipeline_steps,
            data_inspector_verbosity,
            data_inspector_prefix='raw input data',
            data_inspector_name='data_inspector_raw_input',
        )

    # Replaces miscellaneous missing values with expected values (np.nan)
    format_missing_data = FormatMissingData()
    sklearn_Pipeline_steps.append(
        ('format_missing_data', format_missing_data)
    )

    # Make data inspector transformer to check out data
    # after format missing values
    if pipeline_inspect_data_verbosity > 2:
        sklearn_Pipeline_steps = append_data_inspector(
            sklearn_Pipeline_steps,
            data_inspector_verbosity,
            data_inspector_prefix='raw data after format missing values',
            data_inspector_name='data_inspector_post_format_missing_data',
        )

    # Impute
    sklearn_Pipeline_steps.extend(sklearn_pipeline_impute_steps)
    if pipeline_inspect_data_verbosity > 2:
        sklearn_Pipeline_steps = append_data_inspector(
            sklearn_Pipeline_steps,
            data_inspector_verbosity,
            data_inspector_prefix='data after imputation',
            data_inspector_name='data_inspector_post_impute',
        )

    # Encode
    sklearn_Pipeline_steps.extend(sklearn_pipeline_encode_steps)
    if pipeline_inspect_data_verbosity > 2:
        sklearn_Pipeline_steps = append_data_inspector(
            sklearn_Pipeline_steps,
            data_inspector_verbosity,
            data_inspector_prefix='data after encoding',
            data_inspector_name='data_inspector_post_encode',
        )

    # Transform (per-input)
    sklearn_Pipeline_steps.extend(sklearn_pipeline_transform_steps)
    if pipeline_inspect_data_verbosity > 2:
        sklearn_Pipeline_steps = append_data_inspector(
            sklearn_Pipeline_steps,
            data_inspector_verbosity,
            data_inspector_prefix='data after transforms',
            data_inspector_name='data_inspector_post_transform',
        )

    # Step to drop raw inputs not for model
    # (these inputs were used to calculate features,
    # but are not themselves features for the model)
    drop_input_cols = []
    for model_input in inputs_params:
        try:
            if inputs_params[model_input]['drop_before_model']:
                drop_input_cols.append(model_input)
        except KeyError:
            continue
    drop_inputs = DropInputsTransfomer(drop_input_cols)
    sklearn_Pipeline_steps.append(
        ('drop_inputs', drop_inputs)
    )

    # Make data inspector transformer to check out data going into model
    if pipeline_inspect_data_verbosity > 0:
        sklearn_Pipeline_steps = append_data_inspector(
            sklearn_Pipeline_steps,
            data_inspector_verbosity,
            data_inspector_prefix='calculated features provided to model',
            data_inspector_name='data_inspector_pre_model',
        )

    # Model itself
    if model_name == 'PassthroughEstimator':
        model = PassthroughEstimator(**model_params['model_params'])

    elif model_name == 'RandomForestRegressor':
        if model_params['GridSearchCV']:
            print('Performing CV on Grid Search Parameters: ')
            param_grid = model_params['model_params']
            print(param_grid)
            model = GridSearchCV(RandomForestRegressor(),param_grid = param_grid,**model_params['CV_params'])
        else:
            model = RandomForestRegressor(**model_params['model_params'])

    elif model_name == 'XGBRegressor':
        if model_params['GridSearchCV']:
            print('Performing CV on Grid Search Parameters: ')
            param_grid = model_params['model_params']
            print(param_grid)
            model = GridSearchCV(XGBRegressor(),param_grid = param_grid,**model_params['CV_params'])
        else:
            model = XGBRegressor(**model_params['model_params'])

    elif model_name == 'Lasso':
        model = Lasso(**model_params['model_params'])
    else:
        raise ValueError('unknown model type {}'.format(model_name))
    # If model predicts residual of passthrough estimator, wrap it
    if 'predict_residual_of_input' in model_params:
        model_predict_residuals = PassthroughResidualEstimator(
            model,
            model_params['predict_residual_of_input'],
        )
        sklearn_Pipeline_steps.append((
            'model_predict_residuals', model_predict_residuals
        ))
    else:
        sklearn_Pipeline_steps.append((
            'model', model
        ))

    # Make pipeline
    pipeline = sklearn_Pipeline(
        steps=sklearn_Pipeline_steps,
    )

    # Add wrapper to skip model and return nan when core features missing,
    # and target values not satisfying the defined rules
    filter_pipeline = FilterPipeline(
        core_pipeline=pipeline,
        default_response=np.nan,
        print_debug=True,
        default_score_behavior='model',
    )
    filter_pipeline = add_rules_to_filter_pipeline(
        filter_pipeline,
        inputs_params,
        target_params,
    )

    # Add wrapper to convert predicted seconds into predicted datetime
    overall_pipeline = AddDurationToTimestampPipeline(
        core_pipeline=filter_pipeline,
        base_timestamp_col_name='timestamp',
        core_prediction_units='seconds',
    )

    # Train pipeline
    tic = time.time()
    overall_pipeline.fit(
        data.loc[
            (data.group == 'train') & (data.train_sample),
            [i for i in inputs_params.keys() if i in data.columns]
        ],
        data.loc[
            (data.group == 'train') & (data.train_sample),
            target_params['name']
        ],
    )
    toc = time.time()

    log = logging.getLogger(__name__)
    log.info(
        'training actual ON {} model with {} samples '
        'took {:.1f} minutes'.format(
            model_name,
            ((data.group == 'train') & (data.train_sample)).sum(),
            (toc - tic) / 60
        )
    )

    # Init MLflow run for this model
    run_id = init_mlflow_run(
        mlflow_params,
        mlflow_experiment_id,
    )

    # Log trained model, run parameters, and some run metrics
    with mlflow.start_run(run_id=run_id) as active_run:
        mlf_sklearn.log_model(
            sk_model=overall_pipeline,
            artifact_path='model',
            conda_env=add_environment_specs_to_conda_file()
        )
        # log best parameters found by CV
        if (model_name=='RandomForestRegressor' or model_name=='XGBRegressor') and model_params['GridSearchCV']:
            print("best parameters found by CV")

            if 'predict_residual_of_input' in model_params:
                df_grid_search = pd.DataFrame(overall_pipeline.core_pipeline.core_pipeline.named_steps['model_predict_residuals'].core_model.cv_results_)
                best_params = overall_pipeline.core_pipeline.core_pipeline.named_steps['model_predict_residuals'].core_model.best_params_
            else:
                df_grid_search = pd.DataFrame(overall_pipeline.core_pipeline.core_pipeline.named_steps['model'].cv_results_)
                best_params = overall_pipeline.core_pipeline.core_pipeline.named_steps['model'].best_params_

            print(best_params)
            file_path_name = os.path.join(
            'data',
            '07_model_output',
            'grid_search.csv')
            df_grid_search.to_csv(file_path_name,index=False)
            mlflow.log_artifact(file_path_name)
            os.remove(file_path_name)
            mlflow.log_param('best CV parameters', best_params)


        # log core features
        core_features = [n for n in inputs_params if inputs_params[n]["core"] == True]
        mlflow.log_param("core_features", core_features)
        # Set tags
        mlflow.set_tag('airport_icao', globals['airport_icao'])
        # Record modeler name 
        mlflow.log_param('model_name', model_name)
        # Log model parameters one at a time so that character limit is
        # 500 instead of 250
        for key, value in model_params.items():
            mlflow.log_param(key, value)
        # Still hoping we can do all the globals in one shot
        mlflow.log_params(globals)
        # Log training time
        mlflow.log_metric(
            'training_time_minutes',
            ((toc - tic) / 60),
        )
        # Log number of samples used to train
        mlflow.log_metric(
            'training_num_samples',
            ((data.group == 'train') & (data.train_sample)).sum(),
        )
        mlflow.log_metric(
            'testing_num_samples',
            (data.group == 'test').sum(),
        )
        mlflow_run_id = active_run.info.run_id

    return overall_pipeline, mlflow_run_id


def train_models(
    data: pd.DataFrame,
    sklearn_pipeline_impute_steps: List[Tuple],
    sklearn_pipeline_encode_steps: List[Tuple],
    sklearn_pipeline_transform_steps: List[Tuple],
    models_params: Dict[str, Any],
    inputs_params: Dict[str, Any],
    inputs_params_train_only: Dict[str, Any],
    target_params: Dict[str, Any],
    global_params: Dict[str, Any],
    mlflow_params: Dict[str, Any],
    mlflow_experiment_id: int,
    pipeline_inspect_data_verbosity: int = 0,
    data_inspector_verbosity: int = 0,
) -> Dict[str, sklearn_Pipeline]:
    model_pipelines = {}
    model_mlflow_run_ids = {}

    for model_name, model_params in models_params.items():
        model_pipelines[model_name], model_mlflow_run_ids[model_name] =\
            train_model(
                data,
                sklearn_pipeline_impute_steps,
                sklearn_pipeline_encode_steps,
                sklearn_pipeline_transform_steps,
                model_name,
                model_params,
                inputs_params,
                inputs_params_train_only,
                target_params,
                global_params,
                mlflow_params,
                mlflow_experiment_id,
                pipeline_inspect_data_verbosity,
                data_inspector_verbosity,
            )

    return model_pipelines, model_mlflow_run_ids


def predict_models(
    model_pipelines: Dict[str, sklearn_Pipeline],
    data: pd.DataFrame,
    inputs_params: Dict[str, Any],
    model_mlflow_run_ids: Dict[str, str],
) -> pd.DataFrame:
    for model_name, model_pipeline in model_pipelines.items():
        tic = time.time()
        predictions = model_pipeline.predict_core(
            data.loc[
                :,
                [i for i in inputs_params.keys() if i in data.columns]
            ]
        )
        toc = time.time()

        log = logging.getLogger(__name__)
        log.info(
            'predicting {} samples with actual ON {} model '
            'took {:.1f} minutes'.format(
                data.shape[0],
                model_name,
                (toc - tic) / 60,
            )
        )

        # Report prediction time to MLflow as a metric
        with mlflow.start_run(
            run_id=model_mlflow_run_ids[model_name]
        ) as active_run:
            # Log prediction time
            mlflow.log_metric(
                'prediction_time_avg_seconds',
                ((toc - tic) / data.shape[0]),
            )

        # load predictions into data
        data['pred_' + model_name] = predictions

    return data


def report_model_metrics(
    data: pd.DataFrame,
    target: str,
    metrics_params: Dict[str, Any],
    y_pred: str,
    group_values: list = ['train', 'test'],
    name_prefix: str = '',
) -> None:
    metrics_dict = {
        metric_name: METRIC_NAME_TO_FUNCTION_DICT[metric_name]
        for metric_name in [*metrics_params]
    }

    # Log the accuracy of the model
    log = logging.getLogger(__name__)

    for g in group_values:
        log.info(
            'metrics based on {} ({} non-null) predictions '
            'on data in group {}'.format(
                (data.group == g).sum(),
                ((data.group == g) & (data[y_pred].notnull())).sum(),
                g,
            )
        )

    evaluation_df = evaluate_predictions(
        data[data.group.isin(group_values)],
        y_true=target,
        y_pred=y_pred,
        metrics_dict=metrics_dict,
    )

    if 'percent_within_n' in metrics_params:
        percent_within_n_df = calc_percent_within_n_df(
            data[data.group.isin(group_values)],
            y_true=target,
            y_pred=y_pred,
            ns=metrics_params['percent_within_n'].get('ns', [10, 30, 60]),
        )

        evaluation_df = evaluation_df.join(percent_within_n_df)

    for metric_name in [*evaluation_df.columns]:
        log.info('metric {}:'.format(name_prefix + metric_name))
        for group in [v for v in data.group.unique() if v in group_values]:
            log.info('{} group: {}'.format(
                group,
                evaluation_df.loc[group, metric_name]
            ))
            mlflow.log_metric(
                name_prefix + metric_name + '_' + group,
                evaluation_df.loc[group, metric_name]
            )


def report_models_performance_metrics(
    data: pd.DataFrame,
    target: str,
    models_params: Dict[str, Any],
    metrics_params: Dict[str, Any],
    model_mlflow_run_ids: Dict[str, str],
) -> None:
    log = logging.getLogger(__name__)

    for model_name, _ in models_params.items():
        # Restart the MLflow run we used when training this model
        with mlflow.start_run(
            run_id=model_mlflow_run_ids[model_name]
        ) as active_run:

            log.info('reporting on evaluation of model {}'.format(model_name))

            report_model_metrics(
                data,
                target,
                metrics_params,
                'pred_' + model_name,
            )


def log_median_abs_error_lookahead_plot(
    errors_lookaheads: pd.DataFrame,
    model_name: str,
    lookahead_column: str,
    suffix: str = '',
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    max_marker_size = 10 * (mpl.rcParams['lines.markersize'] ** 2)
    marker_size_multiplier = (
        max_marker_size /
        errors_lookaheads['num_samples'].max()
    )

    ax.scatter(
        errors_lookaheads['bin_middle']/60,
        errors_lookaheads['median_absolute_error']/60,
        color='blue',
        label='median absolute error',
        marker='o',
        alpha=0.5,
        s=marker_size_multiplier*errors_lookaheads['num_samples'],
    )

    ax.set_xlim(
        max(errors_lookaheads['bin_upper']/60) + 5,
        min(errors_lookaheads['bin_lower']/60,) - 5
    )
    ax.set_ylim(
        -0.5,
        15,  # errors_lookaheads['median_absolute_error'].max()/60 + 1
    )

    ax.set_ylabel(
        'median absolute error\n' +
        'in {}\n'.format(model_name) +
        'prediction\n'
        '[minutes]',
        rotation=0,
    )
    ax.yaxis.set_label_coords(-0.15, 0.4)

    ax.set_xticks(
        np.flip(np.hstack((
            errors_lookaheads['bin_lower'].values/60,
            errors_lookaheads['bin_upper'].values[-1]/60
        )))
    )
    ax.set_xlabel(
        'lookahead time to {}\n'.format(lookahead_column) +
        '[minutes]'
    )

    ax.grid(zorder=0)

    for direction in ['left', 'right', 'top', 'bottom']:
        # hides borders
        ax.spines[direction].set_visible(False)

    ax.tick_params(axis='x', direction='out', length=3, width=1)
    ax.tick_params(axis='y', direction='out', length=3, width=1)

    plt.tight_layout()

    file_path_name = os.path.join(
        'data',
        '07_model_output',
        'median_abs_error_{}_lookahead_to_{}.png'.format(
            suffix,
            lookahead_column,
        ),
    )
    plt.savefig(file_path_name)
    mlflow.log_artifact(file_path_name)
    os.remove(file_path_name)


def log_error_distr_lookahead_plot(
    errors_lookaheads: pd.DataFrame,
    model_name: str,
    lookahead_column: str,
    suffix: str = '',
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(
        errors_lookaheads['bin_middle']/60,
        errors_lookaheads['q0.05']/60,
        errors_lookaheads['q0.95']/60,
        facecolor='blue',
        edgecolor='blue',
        alpha=0.15,
        label='5th to 95th percentile'
    )
    ax.fill_between(
        errors_lookaheads['bin_middle']/60,
        errors_lookaheads['q0.25']/60,
        errors_lookaheads['q0.75']/60,
        facecolor='blue',
        edgecolor='blue',
        alpha=0.5,
        label='25th to 75th percentile'
    )
    ax.plot(
        errors_lookaheads['bin_middle']/60,
        errors_lookaheads['q0.5']/60,
        color='blue',
        label='median',
        marker='o',
    )
    ax.hlines(
        0,
        min(errors_lookaheads['bin_lower']/60),
        max(errors_lookaheads['bin_upper']/60),
        color='black',
        linewidth=1,
        linestyle='--',
        alpha=1,
    )

    ax.set_xlim(
        max(errors_lookaheads['bin_upper']/60) + 1,
        min(errors_lookaheads['bin_lower']/60) - 1
    )
    ax.set_ylim(
        -16,
        16
    )

    ax.set_ylabel(
        'error\n' +
        '(time to actual ON -\n' +
        'time to\n' +
        '{}\n'.format(model_name) +
        'prediction)\n' +
        '[minutes]',
        rotation=0
    )
    ax.yaxis.set_label_coords(-0.15, 0.4)

    ax.set_xticks(
        np.flip(np.hstack((
            errors_lookaheads['bin_lower'].values/60,
            errors_lookaheads['bin_upper'].values[-1]/60
        )))
    )
    ax.set_xlabel(
        'lookahead time to {}\n'.format(lookahead_column) +
        '[minutes]'
    )

    ax.grid(zorder=0)

    for direction in ['left', 'right', 'top', 'bottom']:
        # hides borders
        ax.spines[direction].set_visible(False)

    ax.legend(loc='best')

    ax.tick_params(axis='x', direction='out', length=3, width=1)
    ax.tick_params(axis='y', direction='out', length=3, width=1)

    plt.tight_layout()

    file_path_name = os.path.join(
        'data',
        '07_model_output',
        'error_distr_{}_lookahead_to_{}.png'.format(
            suffix,
            lookahead_column,
        ),
    )
    plt.savefig(file_path_name)
    mlflow.log_artifact(file_path_name)
    os.remove(file_path_name)


def log_scatter_plot(
    data: pd.DataFrame,
    target: str,
    model_name: str,
    model_params: Dict[str, Any],
    suffix: str = '',
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.scatter(
        data[target]/60,
        data['pred_{}'.format(model_name)]/60,
        s=1,
        alpha=0.1,
    )

    xlim = ax.get_ylim()
    ylim = ax.get_xlim()
    minlim = min(xlim[0], ylim[0])
    maxlim = max(xlim[1], ylim[1])
    ax.set_ylim(minlim, maxlim)
    ax.set_xlim(maxlim, minlim)
    ax.axis('equal')

    ax.plot(
        [minlim, maxlim],
        [minlim, maxlim],
        color='gray',
        linestyle='--',
        alpha=0.5,
    )

    ax.set_ylabel('{}\npredicted time to landing\n[minutes]'.format(model_name))
    ax.set_xlabel('actual time to landing\n[minutes]')

    ax.grid(zorder=0)

    for direction in ['left', 'right', 'top', 'bottom']:
        # hides borders
        ax.spines[direction].set_visible(False)

    ax.tick_params(axis='x', direction='out', length=3, width=1)
    ax.tick_params(axis='y', direction='out', length=3, width=1)

    plt.tight_layout()

    file_path_name = os.path.join(
        'data',
        '07_model_output',
        'scatter_{}.png'.format(suffix),
    )
    plt.savefig(file_path_name)
    mlflow.log_artifact(file_path_name)
    os.remove(file_path_name)


def log_model_performance_artifacts(
    data: pd.DataFrame,
    target: str,
    model_name: str,
    model_params: Dict[str, Any],
    groups: List[str] = ['train', 'test'],
) -> None:
    for group in groups:
        # Scatter plot
        log_scatter_plot(
            data[data.group == group],
            target,
            model_name,
            model_params,
            suffix=group,
        )

        # TODO: general mean/median APE error histogram

        # Residual distributions for use in plotting
        resid_distr_to_actual_df = residual_distribution_summary_lookahead(
            data[data.group == group],
            target,
            'pred_{}'.format(model_name),
            lookahead_column=target,
        )
        resid_distr_to_pred_df = residual_distribution_summary_lookahead(
            data[data.group == group],
            target,
            'pred_{}'.format(model_name),
            lookahead_column='pred_{}'.format(model_name),
        )

        # Error boxplot as function of look-ahead to actual ON time
        log_error_distr_lookahead_plot(
            resid_distr_to_actual_df,
            model_name,
            lookahead_column=target,
            suffix=group,
        )

        # Error boxplot as function of look-ahead to predicted ON time
        log_error_distr_lookahead_plot(
            resid_distr_to_pred_df,
            model_name,
            lookahead_column='pred_{}'.format(model_name),
            suffix=group,
        )

        # Median absolute error as function of look-ahead to actual ON time
        log_median_abs_error_lookahead_plot(
            resid_distr_to_actual_df,
            model_name,
            lookahead_column=target,
            suffix=group,
        )

        # Median absolute error as function of look-ahead to predicted ON time
        log_median_abs_error_lookahead_plot(
            resid_distr_to_actual_df,
            model_name,
            lookahead_column='pred_{}'.format(model_name),
            suffix=group,
        )

        # TODO: Percent w/in threshold as function of look-ahead time


def log_models_performance_artifacts(
    data: pd.DataFrame,
    target: str,
    models_params: Dict[str, Any],
    model_mlflow_run_ids: Dict[str, str],
) -> None:
    log = logging.getLogger(__name__)

    for model_name, model_params in models_params.items():
        # Restart the MLflow run we used when training this model
        with mlflow.start_run(
            run_id=model_mlflow_run_ids[model_name]
        ) as active_run:

            log.info('logging performance artifacts for model {}'.format(
                model_name
            ))

            log_model_performance_artifacts(
                data,
                target,
                model_name,
                model_params,
            )


def log_models_sample_data(
    data: pd.DataFrame,
    models_params: Dict[str, Any],
    inputs_params: Dict[str, Any],
    target_params: Dict[str, Any],
    model_mlflow_run_ids: Dict[str, str],
    data_set_size: int = 10,
    groups: List[str] = ['train', 'test'],
) -> None:
    log = logging.getLogger(__name__)

    # Create sample data
    input_data = data.loc[
        data.group.isin(groups),
        [i for i in inputs_params.keys() if i in data.columns]
    ]
    sampled_index = np.random.choice(
        input_data.index,
        size=data_set_size,
        replace=False,
    )
    input_data = input_data.loc[sampled_index, :]
    target_data = data.loc[
        sampled_index,
        target_params['name']
    ]
    # Drop 2nd level in input_data index data set
    input_data = input_data.droplevel(1)

    input_data_path_name = os.path.join(
        'data',
        '05_model_input',
        'sample_input_data.csv',
    )
    target_data_path_name = os.path.join(
        'data',
        '05_model_input',
        'sample_target_data.csv',
    )

    input_data.to_csv(input_data_path_name)
    target_data.to_csv(target_data_path_name)
    
    # get features to log in mlflow
    input_features = list(input_data.columns)
    
    for model_name, _ in models_params.items():
        # Restart the MLflow run we used when training this model
        with mlflow.start_run(
            run_id=model_mlflow_run_ids[model_name]
        ) as active_run:

            log.info('logging sample data for model {}'.format(
                model_name
            ))

            mlflow.log_artifact(input_data_path_name, 'model')
            mlflow.log_artifact(target_data_path_name, 'model')
            mlflow.log_param("features", input_features)

    os.remove(input_data_path_name)
    os.remove(target_data_path_name)


def tag_with_arr_rwy_model_uri(
    model_mlflow_run_ids: Dict[str, str],
    arr_rwy_model_uri: str,
) -> None:
    for model_name, model_run_id in model_mlflow_run_ids.items():
        with mlflow.start_run(run_id=model_run_id) as active_run:
            mlflow.log_param(
                'arr_rwy_model_uri',
                arr_rwy_model_uri
            )
