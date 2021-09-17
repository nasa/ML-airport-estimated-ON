"""Nodes for querying and saving data sets.
"""

from typing import Any, Dict, List
from kedro.io import DataCatalog, Version
from kedro.extras.datasets.pandas import CSVDataSet

import logging

import pandas as pd
import numpy as np


def query_save_version(
    data: pd.DataFrame,
    catalog_name: str,
    file_name: str,
    folder='01_raw',
    versioned=False,
):
    """Saves results of DB query to an @CSV version of the data set

    Note: Assumed that data comes from {catalog_name}_data_set@DB and then
    save resulting CSV to data/{folder}/{file_name}_data_set@CSV,
    registering it in the data catalog as {catalog_name}_data_set@CSV
    """
    if versioned:
        version = Version(
            load=None,
            save=None,
        )
    else:
        version = None

    data_set_CSV = CSVDataSet(
        filepath="data/{}/{}_data_set.csv".format(
            folder,
            file_name
        ),
        save_args={"index": False},
        version=version,
    )
    dc = DataCatalog({"{}_data_set@CSV".format(catalog_name): data_set_CSV})

    dc.save("{}_data_set@CSV".format(catalog_name), data)


def query_save_version_TBFM(
    data: pd.DataFrame,
    params_globals: Dict[str, Any],
):
    log = logging.getLogger(__name__)

    query_save_version(
        data,
        "TBFM",
        params_globals['airport_icao'] + '_' +
        str(params_globals['start_time']) + '_' +
        str(params_globals['end_time']) +
        ".TBFM",
    )

    log.info('done with query and save of TBFM data for {}'.format(
        params_globals['airport_icao']
    ))


def query_save_version_TFM_track(
    data: pd.DataFrame,
    params_globals: Dict[str, Any],
):
    log = logging.getLogger(__name__)

    query_save_version(
        data,
        "TFM_track",
        params_globals['airport_icao'] + '_' +
        str(params_globals['start_time']) + '_' +
        str(params_globals['end_time']) +
        ".TFM_track",
    )

    log.info('done with query and save of TFM track data for {}'.format(
        params_globals['airport_icao']
    ))


def query_save_version_MF_TFM(
    data: pd.DataFrame,
    params_globals: Dict[str, Any],
):
    log = logging.getLogger(__name__)

    query_save_version(
        data,
        "MF_TFM",
        params_globals['airport_icao'] + '_' +
        str(params_globals['start_time']) + '_' +
        str(params_globals['end_time']) +
        ".MF_TFM",
    )

    log.info('done with query and save of MATM flight TFM data for {}'.format(
        params_globals['airport_icao']
    ))


def query_save_version_MFS(
    data: pd.DataFrame,
    params_globals: Dict[str, Any],
):
    log = logging.getLogger(__name__)

    query_save_version(
        data,
        "MFS",
        params_globals['airport_icao'] + '_' +
        str(params_globals['start_time']) + '_' +
        str(params_globals['end_time']) +
        ".MFS",
    )

    log.info('done with query and save of MATM flight summary data for {}'.format(
        params_globals['airport_icao'],
    ))


def query_save_version_first_position(
    data: pd.DataFrame,
    params_globals: Dict[str, Any],
):
    log = logging.getLogger(__name__)

    query_save_version(
        data,
        "first_position",
        params_globals['airport_icao'] + '_' +
        str(params_globals['start_time']) + '_' +
        str(params_globals['end_time']) +
        ".first_position",
    )

    log.info('done with query and save of first position data for {}'.format(
        params_globals['airport_icao']
    ))


def query_save_version_runways(
    data: pd.DataFrame,
    params_globals: Dict[str, Any],
):
    log = logging.getLogger(__name__)

    query_save_version(
        data,
        "runways",
        params_globals['airport_icao'] + '_' +
        str(params_globals['start_time']) + '_' +
        str(params_globals['end_time']) +
        ".runways",
    )

    log.info('done with query and save of arrival runways and times data for {}'.format(
        params_globals['airport_icao']
    ))


def query_save_version_landing_position(
    data: pd.DataFrame,
    params_globals: Dict[str, Any],
):
    log = logging.getLogger(__name__)

    query_save_version(
        data,
        "landing_position",
        params_globals['airport_icao'] + '_' +
        str(params_globals['start_time']) + '_' +
        str(params_globals['end_time']) +
        ".landing_position",
    )

    log.info('done with query and save of arrival landing position data for {}'.format(
        params_globals['airport_icao']
    ))


def query_save_version_stand_scheduled_times(
    data: pd.DataFrame,
    params_globals: Dict[str, Any],
):
    log = logging.getLogger(__name__)

    query_save_version(
        data,
        "stand_scheduled_times",
        params_globals['airport_icao'] + '_' +
        str(params_globals['start_time']) + '_' +
        str(params_globals['end_time']) +
        ".stand_scheduled_times",
    )

    log.info('done with query and save of scheduled stand times data for {}'.format(
        params_globals['airport_icao']
    ))
