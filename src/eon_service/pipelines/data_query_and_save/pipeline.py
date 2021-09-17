"""Pipeline for actual ON prediction data query and save
"""

import os
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *


def create_pipelines(**kwargs):

    dqs_pipeline = Pipeline(
        [
            node(
                func=query_save_version_TBFM,
                inputs=[
                    "TBFM_data_set@DB",
                    "params:globals",
                ],
                outputs=None,
                name='query_save_version_TBFM',
            ),
            node(
                func=query_save_version_TFM_track,
                inputs=[
                    "TFM_track_data_set@DB",
                    "params:globals",
                ],
                outputs=None,
                name='query_save_version_TFM_track',
            ),
            node(
                func=query_save_version_MF_TFM,
                inputs=[
                    "MF_TFM_data_set@DB",
                    "params:globals",
                ],
                outputs=None,
                name='query_save_version_MF_TFM',
            ),
            node(
                func=query_save_version_MFS,
                inputs=[
                    "MFS_data_set@DB",
                    "params:globals",
                ],
                outputs=None,
                name='query_save_version_MFS',
            ),
            node(
                func=query_save_version_first_position,
                inputs=[
                    "first_position_data_set@DB",
                    "params:globals",
                ],
                outputs=None,
                name='query_save_version_first_position',
            ),
            node(
                func=query_save_version_runways,
                inputs=[
                    "runways_data_set@DB",
                    "params:globals",
                ],
                outputs=None,
                name='query_save_version_runways',
            ),
            node(
                func=query_save_version_landing_position,
                inputs=[
                    'landing_position_data_set@DB',
                    'params:globals',
                ],
                outputs=None,
                name='query_save_landing_position',
            ),
        ],
        tags='dqs',
    )

    return dqs_pipeline
