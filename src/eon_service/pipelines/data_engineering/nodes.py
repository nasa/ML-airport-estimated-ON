from typing import Any, Dict, List
import logging

import numpy as np
import pandas as pd

from datetime import datetime, timedelta, date

from kedro.extras.datasets.pickle import PickleDataSet
import os
import pickle

import mlflow
import mlflow.sklearn


def df_passback(
    data: pd.DataFrame,
) -> pd.DataFrame:
    return data


def df_min_per_gufi(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Grab first row per gufi & reset index
    """
    data = data.groupby('gufi').min()
    data = data.reset_index()
    return data


def df_join(
    data_0: pd.DataFrame,
    data_1: pd.DataFrame,
    join_kwargs: Dict[str, Any] = {},
) -> pd.DataFrame:
    """Join data_1 on to data_0, per join_kwargs.
    join_kwargs can be specified per join in parameters.yml.
    """
    return data_0.join(data_1, **join_kwargs)


def set_index(
    data: pd.DataFrame,
    new_index='gufi',
    set_index_kwargs: Dict[str, Any] = {},
) -> pd.DataFrame:
    """Set index to new_index, per set_index_kwargs.
    set_index_kwargs can be specified in parameters.yml.
    """
    data = data.set_index(new_index, **set_index_kwargs)

    return data


def add_col_name_suffix(
    data: pd.DataFrame,
    col_name_suffix: str,
    except_cols: List[str] = ['gufi', 'timestamp'],
) -> pd.DataFrame:
    """Add a col_name_suffix to names of columns in data, excepting except_cols
    """
    data = data.rename(
        columns={
            col: col + col_name_suffix
            for col in data.columns
            if col not in except_cols
        },
    )

    return data


def filter_rows_missing_columns(
    data: pd.DataFrame,
    col_names: List[str] = ['arrival_runway_actual_time_via_surveillance'],
) -> pd.DataFrame:
    """Filter rows if any of columns in col_names list are missing.
    """
    data = data.dropna(subset=col_names)

    return data


def drop_cols_not_in_inputs(
    data: pd.DataFrame,
    inputs: Dict[str, Any],
    inputs_train_only: List[str] = [],
    except_cols: List[str] = ['arrival_runway_actual_time_via_surveillance', 'train_sample'],
) -> pd.DataFrame:
    """Drop column not specified as "inputs"
    (keys in inputs dict from parameters.yml).
    Optionally also specify some columns needed for training but nothing else.
    Don't drop excep_cols.
    Re-order columns per inputs ordering.
    """
    # Always keep gufi too
    keep_cols = ['gufi'] + [*inputs.keys()] + inputs_train_only + except_cols
    col_list = [*data.columns]

    data = data.drop(
        columns=[
            col for col in col_list
            if col not in keep_cols
        ],
    )

    # Reorder per input specification
    data = data[[col for col in keep_cols if col in data.columns]]

    return data


def drop_timestamp_col(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Drop the timestamp column
    """
    return data.drop(columns='timestamp')


def sort_by_and_create_multiindex(
    data: pd.DataFrame,
    sort_and_multiindex_cols: List[str] = ['gufi', 'timestamp'],
    set_index_kwargs: Dict[str, Any] = {'drop': False},
) -> pd.DataFrame:
    """Sort by columns, then create multi-index with them,
    per set_index_kwargs, which can be specified in parameters.yml
    """
    data = data.sort_values(sort_and_multiindex_cols)
    data = data.set_index(sort_and_multiindex_cols, **set_index_kwargs)

    return data


def de_dup(
    data: pd.DataFrame,
    drop_duplicates_kwargs: Dict[str, Any] = {},
) -> pd.DataFrame:
    """Drop duplicate rows
    """
    return data.drop_duplicates(**drop_duplicates_kwargs)


def de_dup_index(
    data: pd.DataFrame,
    drop_duplicated_index_kwargs: Dict[str, Any] = {},
) -> pd.DataFrame:
    """Drop duplicate indices."""
    keep_idx = ~data.index.duplicated(**drop_duplicated_index_kwargs)

    return data[keep_idx]


def de_dup_stand_scheduled_times(
    data: pd.DataFrame
) -> pd.DataFrame:
    """Sort by stand times, then grab first per gufi
    """
    # Sort by dep stand time timestamp, then grab first per gufi
    data = data.sort_values('departure_stand_scheduled_time_timestamp')
    firsts_dep = data.loc[
        data.departure_stand_scheduled_time.notnull(),
        [
            'gufi',
            'departure_stand_scheduled_time_timestamp',
            'departure_stand_scheduled_time',
        ]
    ].groupby('gufi').first()

    # Sort by arr stand time timestamp, then grab first per gufi
    data = data.sort_values('arrival_stand_scheduled_time_timestamp')
    firsts_arr = data.loc[
        data.arrival_stand_scheduled_time.notnull(),
        [
            'gufi',
            'arrival_stand_scheduled_time_timestamp',
            'arrival_stand_scheduled_time',
        ]
    ].groupby('gufi').first()

    # Join results and send out
    return firsts_arr.join(firsts_dep)


def start_tv_df(
    ntv_df: pd.DataFrame,
    tv_timestep: str = '30s',
) -> pd.MultiIndex:
    """Creates a time varying data set with a row every tv_timestep per gufi,
    starting at gufi departure_runway_actual_time or time_first_tracked
    and ending just after arrival_runway_actual_time_via_surveillance
    """
    tv_df_dict = {
        'gufi': [],
        'timestamp': [],
    }

    for gufi in ntv_df.index:
        if ntv_df.loc[gufi, 'departure_runway_actual_time'] is not pd.NaT:
            start_timestamp = ntv_df.loc[gufi, 'departure_runway_actual_time']
        elif ntv_df.loc[gufi, 'time_first_tracked'] is not pd.NaT:
            start_timestamp = ntv_df.loc[gufi, 'time_first_tracked']
        else:
            continue

        gufi_tv_index = pd.date_range(
            start=start_timestamp,
            end=ntv_df.loc[
                gufi,
                'arrival_runway_actual_time_via_surveillance'
            ] + pd.Timedelta(tv_timestep),
            freq=tv_timestep,
        )

        tv_df_dict['timestamp'].extend(gufi_tv_index)
        tv_df_dict['gufi'].extend([gufi] * len(gufi_tv_index))

    tv_df = pd.DataFrame.from_dict(tv_df_dict)

    return tv_df


def sort_timestamp_merge_asof(
    data_0: pd.DataFrame,
    data_1: pd.DataFrame,
    merge_asof_kwargs: Dict[str, Any] = {
        'by': 'gufi',
        'on': 'timestamp',
        'allow_exact_matches': True,
        'direction': 'backward',
    },
) -> pd.DataFrame:
    """Sort values by timestamp, then merge_asof per merge_asof_kwargs,
    which can be specified in parameters.yml.
    """
    data_0 = data_0.sort_values('timestamp')
    data_1 = data_1.sort_values('timestamp')

    data = pd.merge_asof(
        data_0,
        data_1,
        **merge_asof_kwargs,
    )

    return data


def calculate_ATM_date(
    utc_timestamp: datetime,  # tz naive
    local_tz_name: str,
    ATM_date_start_hour_local: int = 4,
) -> date:
    """Take in a tz naive utc_timetamp and find its "ATM date" based on a
    timezone and local hour of day at which "ATM date" starts
    (usually 4am local).
    """
    if utc_timestamp.tz is not None:
        raise(ValueError, 'utc_timestamp is not tz naive')

    local_datetime = utc_timestamp\
        .tz_localize('UTC')\
        .tz_convert(local_tz_name)

    if local_datetime.hour >= ATM_date_start_hour_local:
        ATM_date = local_datetime.date()
    else:
        ATM_date = local_datetime.date() - timedelta(days=1)

    return ATM_date


def add_train_test_group_per_date_random(
    data: pd.DataFrame,
    test_size: float,
    random_seed: int,
    tz_name: str,
    ATM_date_start_hour_local: int = 4,
    datetime_col: str = 'arrival_runway_actual_time_via_surveillance',
) -> pd.DataFrame:
    """Assign rows a train or test group by randomly assigning train or test
    to each ATM date. May help avoid leakage from test into train data sets.
    """
    # Get the ATM dates in the data
    data['ATM_date'] = data.apply(
        lambda row: calculate_ATM_date(
            row[datetime_col],
            tz_name,
            ATM_date_start_hour_local,
        ),
        axis=1,
    )

    ATM_date_set = data['ATM_date'].unique()

    # Build data frame for assignment of groups to dates
    dates_group = pd.DataFrame(
        index=ATM_date_set,
        columns=['group'],
    )

    # Randomly assign train or test group per date
    np.random.seed(random_seed)
    dates_group['uniform_random_sample'] = np.random.uniform(
        size=dates_group.shape[0],
    )
    dates_group['group'] = 'train'
    dates_group.loc[
        (dates_group.uniform_random_sample < test_size),
        'group'
    ] = 'test'
    dates_group = dates_group.drop(columns=['uniform_random_sample'])

    # Join group back onto main data
    data = data.join(
        dates_group,
        on='ATM_date',
    )

    # Drop the ATM date column
    data = data.drop(columns=['ATM_date'])

    return data


def add_train_test_group_by_datetime(
    data: pd.DataFrame,
    test_set_start_time: datetime,
    datetime_col: str = 'arrival_runway_actual_time_via_surveillance',
) -> pd.DataFrame:
    """Assign test group to all dates on or after test_set_start_time,
    others assigned to train group.
    """
    data['group'] = data.apply(
        lambda row: (
            'test' if row[datetime_col] >= test_set_start_time
            else 'train'
        ),
        axis=1,
    )

    return data


def add_train_test_group_per_date(
    data: pd.DataFrame,
    parameters: Dict[str, Any],
) -> pd.DataFrame:
    """Add train & test group assignments per date
    (rather than per gufi or per timestamp)
    """
    log = logging.getLogger(__name__)

    if isinstance(parameters['globals']['test_set_start_time'], datetime):
        # If a datetime test_set_start_time is specified in parameters, use it
        log.info(
            'assigning train & test groups based on test_set_start_time' +
            'in globals.yml'
        )
        data = add_train_test_group_per_date_by_time(
            data,
            parameters['globals']['test_set_start_time'],
        )
    else:
        # If no valid datetime test_set_start_time in parameters,
        # default to random
        log.info(
            'assigning train & test groups randomly ' +
            'and based on test_size and random_seed parameters'
        )
        data = add_train_test_group_per_date_random(
            data,
            parameters['test_size'],
            parameters['random_seed'],
            parameters['globals']['tz_name'],
        )

    return data


def select_tv_train_samples(
    data: pd.DataFrame,
    tv_timestep_fraction_train: float,
    random_seed: int,
    tv_keep_non_train_timesteps: bool = False,
) -> pd.DataFrame:
    """Select a subset of the time-varying samples to use in model training.
    """
    # TODO: adjust to ensure at least one timestamp per gufi
    # TODO: adjust to ensure that don't get timestamps "too close" together
    log = logging.getLogger(__name__)
    log.info('sampling to just use a sub-set ({:.1f}%) of time steps for training'.format(
        tv_timestep_fraction_train*100
    ))

    np.random.seed(random_seed)
    data['uniform_random_sample'] = np.random.uniform(size=data.shape[0])
    data['train_sample'] = (
        data.uniform_random_sample <
        tv_timestep_fraction_train
    )
    data = data.drop(columns=['uniform_random_sample'])

    if not tv_keep_non_train_timesteps:
        tv_samples_before = data.shape[0]
        data = data[data.train_sample]
        tv_samples_after = data.shape[0]

        log.info(
            'removed {} of {} rows ({:.0f}%) of timesteps '.format(
                (tv_samples_before - tv_samples_after),
                tv_samples_before,
                ((tv_samples_before - tv_samples_after)/tv_samples_before)*100,
            ) +
            'to keep only samples used in training'
        )

    return data


def drop_gufis_actual_arrival_time_difference(
    data: pd.DataFrame,
    max_arrival_time_difference_seconds: int = 600,
) -> pd.DataFrame:
    """Drop gufis for which arrival_runway_actual_time_via_surveillance and
    arrival_runway_actual_time are very different for some reason.
    """
    # Add column for difference between
    # arrival_runway_actual_time_via_surveillance
    # and arrival_runway_actual_time
    data['arrival_runway_actual_time_difference'] = (
        data.arrival_runway_actual_time_MFS -
        data.arrival_runway_actual_time_via_surveillance
    ).dt.total_seconds()

    rows_before = data.shape[0]

    # Remove rows when this difference is too large
    # Indicates issue where we might see a data mismatch
    data = data[
        data.arrival_runway_actual_time_difference.abs() <=
        max_arrival_time_difference_seconds
    ]

    rows_after = data.shape[0]

    log = logging.getLogger(__name__)
    log.info(
        'removed {} of {} rows ({:.0f}%) of rows because '.format(
            (rows_before - rows_after),
            rows_before,
            ((rows_before - rows_after)/rows_before)*100
        ) +
        'arrival_runway_actual_time was more than {} seconds '.format(
            max_arrival_time_difference_seconds
        ) +
        'different than arrival_runway_actual_time_via_surveillance'
    )

    # Drop extra column
    data = data.drop(columns=['arrival_runway_actual_time_difference'])

    return data


def drop_gufis_no_points_on_runway(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Drop gufis for which we had no points on runway surveillance to
    aid in actual runway and actual runway time determination.
    Those determinations are less accurate without this runway surveillance.
    """
    rows_before = data.shape[0]

    # Remove rows when this difference is too large
    # Indicates issue where we might see a data mismatch
    data = data[data.points_on_runway.notnull()]
    data = data[data.points_on_runway]

    rows_after = data.shape[0]

    log = logging.getLogger(__name__)
    log.info(
        'removed {} of {} rows ({:.0f}%) of rows because '.format(
            (rows_before - rows_after),
            rows_before,
            ((rows_before - rows_after)/rows_before)*100
        ) +
        'of no surveillance points on runway.'
    )

    return data


def drop_gufis_at_landing_time_difference(
    data: pd.DataFrame,
    max_at_landing_time_difference_seconds: int = 600,
) -> pd.DataFrame:
    """Drop gufis for which arrival_runway_actual_time_via_surveillance and
    position_timestamp_at_landing are very different for some reason.
    """
    # Add column for difference between
    # arrival_runway_actual_time_via_surveillance
    # and arrival_runway_actual_time
    data['at_landing_time_difference'] = (
        data.position_timestamp_at_landing -
        data.arrival_runway_actual_time_via_surveillance
    ).dt.total_seconds()

    rows_before = data.shape[0]

    # Remove rows when this difference is too large
    # Indicates issue where we might see a data mismatch
    data = data[
        data.at_landing_time_difference.abs() <= 
        max_at_landing_time_difference_seconds
    ]

    rows_after = data.shape[0]

    log = logging.getLogger(__name__)
    log.info(
        'removed {} of {} rows ({:.0f}%) of rows because '.format(
            (rows_before - rows_after),
            rows_before,
            ((rows_before - rows_after)/rows_before)*100
        ) +
        'position_timestamp_at_landing was more than {} seconds '.format(
            max_at_landing_time_difference_seconds
        ) +
        'different than arrival_runway_actual_time_via_surveillance'
    )

    # Drop extra column
    data = data.drop(columns=['at_landing_time_difference'])

    return data


def filter_times_before_time_first_tracked(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Remove samples with timestamp prior to time_first_tracked.
    """
    data = data[data.time_first_tracked <= data.index.get_level_values(1)]

    return data


def compute_seconds_to_arrival_runway_actual_time(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Compute time from timestamp (index level 1)
    to arrival_runway_actual_time_via_surveillance,
    in seconds.
    """
    data['seconds_to_arrival_runway_actual_time_via_surveillance'] = (
        data['arrival_runway_actual_time_via_surveillance'] -
        data.index.get_level_values(1)
    ).dt.total_seconds()

    return data


def add_empty_predicted_arrival_runway(
    data: pd.DataFrame,
    known_runways: List[str] = None,
) -> pd.DataFrame:
    """Adds empty expected columns for predicted arrival runway.
    If provided known_runways, then create a column per runway with
    the predicted probability.
    Otherwise, just provide a single column for the predicted runway.
    """

    if known_runways is None:
        data['predicted_arrival_runway'] = np.nan
    else:
        for runway in known_runways:
            data['predicted_arrival_runway_' + runway] = np.nan

    return data


def add_empty_num_arrival_runway_best_time(
    data: pd.DataFrame,
    inputs: Dict[str, Any],
) -> pd.DataFrame:
    """Adds empty expected columns for counts of arrivals to the same runway
    (according to TBFM) that have best SWIM ETAs near the best SWIM ETA
    of this arrival.
    Eventually I'll build out the data engineering to compute these,
    but for now this will enable me to build the imputation of these.
    """

    num_arrival_runway_best_time_cols = [
        col for col in [*inputs.keys()]
        if 'num_arrival_runway_best_time' in col
    ]

    for col in num_arrival_runway_best_time_cols:
        data[col] = np.nan

    return data


def log_df_size(
    data: pd.DataFrame,
) -> None:
    """Log size of a data frame
    """
    log = logging.getLogger(__name__)
    log.info('df shape: {} rows x {} columns'.format(
        data.shape[0],
        data.shape[1],
    ))
    log.info('df memory usage in bytes: {}'.format(
        data.memory_usage(deep=True).sum()
    ))


def dropna_core(
    data: pd.DataFrame,
    params_inputs: Dict[str, Any],
):
    """Drop rows with na for any core feature
    """
    rows_before = data.shape[0]

    core_inputs_list = [
        model_input for model_input in params_inputs.keys()
        if params_inputs[model_input]['core']
    ]

    data = data.dropna(
        axis=0,
        how='any',
        subset=core_inputs_list,
    )

    rows_after = data.shape[0]

    log = logging.getLogger(__name__)
    log.info(
        'removed {} of {} rows ({:.0f}%) of rows because '.format(
            (rows_before - rows_after),
            rows_before,
            ((rows_before - rows_after)/rows_before)*100
        ) +
        'missing one or more core features or the target'
    )

    return data


def dropna_col(
    data: pd.DataFrame,
    drop_col: str,
):
    """Drop rows with na for any core feature
    """
    rows_before = data.shape[0]

    data = data.dropna(
        subset=[drop_col]
    )

    rows_after = data.shape[0]

    log = logging.getLogger(__name__)
    log.info(
        'removed {} of {} rows ({:.0f}%) of rows because '.format(
            (rows_before - rows_after),
            rows_before,
            ((rows_before - rows_after)/rows_before)*100
        ) +
        'missing {}'.format(drop_col)
    )

    return data


def prepare_aircraft_class_map(
    aircraft_categories: pd.DataFrame,
    category_col: str = 'category',
) -> Dict[str, str]:
    # prepare aircraft class data
    aircraft_categories.aircraft_type = aircraft_categories.aircraft_type.astype(str)
    aircraft_categories.set_index('aircraft_type', inplace=True)
    aircraft_categories = aircraft_categories.loc[:, [category_col]]
    aircraft_categories = aircraft_categories.squeeze().to_dict()

    return aircraft_categories


def save_aircraft_categories(
    aircraft_categories: Dict[str, str],
    data_folder: str = './data/05_model_input/',
) -> None:
    aircraft_categories_dict = PickleDataSet(
        filepath=data_folder + "aircraft_categories_dict.pkl", backend="pickle")
    aircraft_categories_dict.save(aircraft_categories)


def merge_asof_arr_rwy_data_engred(
    data: pd.DataFrame,
    data_arr_rwy: pd.DataFrame,
    params_globals: Dict[str, Any],
    merge_asof_tolerance: int = 120,
    sort_and_multiindex_cols: List[str] = ['gufi', 'timestamp'],
    set_index_kwargs: Dict[str, Any] = {'drop': False},
) -> pd.DataFrame:

    log = logging.getLogger(__name__)

    log.info(
        'data engineered for arrival runway model ' +
        'has size {}'.format(data_arr_rwy.shape)
    )

    # Compute lookahead so can join in the data_arr_rwy
    data['lookahead'] = (
        data['arrival_runway_best_time'] -
        data['timestamp']
    ).dt.total_seconds()
    # Handle missing lookaheads (these rows likely dropped later anyhow)
    UNREALISTICALLY_LARGE_LOOKAHEAD = 10**6
    data = data.fillna(value={'lookahead': UNREALISTICALLY_LARGE_LOOKAHEAD})
    # Get the data ready for a merge_asof
    data = data.reset_index(level=1, drop=True)
    data = data.reset_index(level=0, drop=True)
    # Rename columns of arr_rwy data so we know which are which
    data_arr_rwy = data_arr_rwy.rename(columns={
        col: (col + '_arr_rwy') for col in data_arr_rwy.columns
        if col not in ['gufi', 'lookahead']
    })
    # do the merge_asof
    data = pd.merge_asof(
        data.sort_values(by='lookahead'),
        data_arr_rwy[data_arr_rwy.lookahead.notnull()]
            .sort_values(by='lookahead'),
        on='lookahead',
        by='gufi',
        tolerance=merge_asof_tolerance,
        direction='forward',  # grab row in data_arr_rwy with larger lookahead
    )
    # Re-set the null lookahead
    data.loc[
        data.lookahead == UNREALISTICALLY_LARGE_LOOKAHEAD,
        'lookahead'
    ] = np.nan

    # Rename 'lookahead' too because only arr_rwy will use it
    # TODO: get the lookahead from data_arr_rwy, rather than the one from data
    # could do this by swapping around which is left and right in merge_asof?
    data = data.rename(columns={'lookahead': 'lookahead_arr_rwy'})

    # Report on merge_asof
    log.info(
        'merge_asof provided engineered data ' +
        'for arrival runway prediction model to ' +
        '{} of {} rows ({:.0f}%) of rows'.format(
            data.lookahead_arr_rwy.notnull().sum(),
            data.shape[0],
            data.lookahead_arr_rwy.notnull().mean()*100,
        )
    )

    # Re-sort data and multi-index like before
    data = data.sort_values(sort_and_multiindex_cols)
    data = data.set_index(sort_and_multiindex_cols, **set_index_kwargs)

    # Remove gufi column
    data = data.drop(columns=['gufi'])

    return data


def load_and_predict_w_arr_rwy_model(
    data: pd.DataFrame,
    params_mlflow: Dict[str, Any],
    arr_rwy_model_uri: str,
):
    # Set up MLflow
    mlflow.set_tracking_uri(params_mlflow['tracking_uri'])

    # Load model
    log = logging.getLogger(__name__)

    log.info(
        'loading arrival runway prediction model ' +
        'with MLflow model URI: {}'.format(arr_rwy_model_uri)
    )

    arr_rwy_model = mlflow.sklearn.load_model(arr_rwy_model_uri)

    # Predict
    arr_rwy_cols = [
        col for col in data.columns
        if '_arr_rwy' in col
    ]

    arr_rwy_pred_df = arr_rwy_model.predict_df(
        data[arr_rwy_cols].rename(columns={
            col: col.split('_arr_rwy')[0] for col in arr_rwy_cols
        })
    )

    data = data.join(
        arr_rwy_pred_df['pred'],
        how='left',
    )
    data = data.rename(columns={'pred': 'predicted_arrival_runway'})

    log.info(
        'arrival runway model provided non-null predictions for ' +
        '{} of {} rows ({:.0f}%) of rows'.format(
            data.predicted_arrival_runway.notnull().sum(),
            data.shape[0],
            data.predicted_arrival_runway.notnull().mean()*100,
        )
    )

    log.info(
        'arrival runway model predictions differ from TBFM predictions ' +
        '{} of {} rows ({:.0f}%) of rows'.format(
            (data.predicted_arrival_runway != data.arrival_runway_tbfm).sum(),
            data.shape[0],
            (
                data.predicted_arrival_runway !=
                data.arrival_runway_tbfm
            ).mean()*100,
        )
    )

    # Drop _arr_rwy columns no longer needed
    data = data.drop(columns=[
        col for col in data.columns
        if '_arr_rwy' in col
    ])

    return data


def impute_predicted_arrival_runway_from_TBFM(
    data: pd.DataFrame,
):
    """Impute the predicted arrival runway with TBFM runway for now
    """
    data['predicted_arrival_runway'] = data['arrival_runway_tbfm']

    return data


def de_save(
    data: pd.DataFrame,
    params_globals: Dict[str, Any],
    data_folder: str = './data/05_model_input/',
) -> None:

    if 'batch_mode' in params_globals:

        # Delete previous runs batch files for airport_icao
        if params_globals['start_time'] == params_globals['batch_mode']['run_start_time']:
            files = os.listdir(data_folder)
            files = [f for f in files if
                     f[0:len(params_globals['airport_icao']) + 1] == params_globals['airport_icao'] + '_']
            for f in files:
                os.remove(data_folder + f)

        # Save current batch
        data_set = PickleDataSet(
            filepath=data_folder + params_globals['airport_icao'] + '_' + str(params_globals['start_time']) \
                     + '_' + str(params_globals['end_time']) + ".de_data_set.pkl", backend="pickle")
        data_set.save(data)

        # Concatenate all data in single file in last iteration, ds pipeline expecting single file
        if params_globals['end_time'] >= params_globals['batch_mode']['run_end_time']:

            files = os.listdir(data_folder)
            files = [f for f in files if f[0:len(params_globals['airport_icao']) + 1] == params_globals['airport_icao'] + '_']

            file_start_dates = [date.fromisoformat(f.split('_')[1]) for f in files]
            idx_sorted = np.argsort(file_start_dates)

            # Load data ordered
            de_data = []
            for idx in idx_sorted:
                with open(data_folder + files[idx], "rb") as f:
                    de_data.append(pickle.load(f))

            # Concatenate all data, keep order and remove duplicates
            de_data = pd.concat(de_data, sort=False)
            # For duplicates, keep "last", the "first" duplicates from previous batch may not include all data due to
            # the end of batch
            de_data = de_data[de_data['timestamp'].duplicated(keep='last') == False]

            # Save data
            data_set = PickleDataSet(
                filepath=data_folder + params_globals['airport_icao'] + ".de_data_set.pkl", backend="pickle")
            data_set.save(de_data)
    else:

        data_set = PickleDataSet(
            filepath=data_folder + params_globals['airport_icao'] + ".de_data_set.pkl", backend="pickle")
        data_set.save(data)
