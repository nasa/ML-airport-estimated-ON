from kedro.pipeline import Pipeline, node, pipeline

from .nodes import *
from data_services.mlflow_utils import *

from data_services.feature_prep_utils import assemble_impute_steps
from data_services.feature_prep_utils import assemble_encode_steps
from data_services.feature_prep_utils import assemble_transform_steps


def create_pipelines(**kwargs):
    train_test_pipeline = Pipeline(
        [
            node(
                func=init_mlflow,
                inputs='parameters',
                outputs='mlflow_experiment_id',
                name='init_mlflow',
            ),
            node(
                func=assemble_impute_steps,
                inputs=[
                    'params:inputs',
                    'params:pipeline_inspect_data_verbosity',
                    'params:data_inspector_verbosity',
                ],
                outputs='sklearn_pipeline_impute_steps',
                name='assemble_impute_steps',
            ),
            node(
                func=assemble_encode_steps,
                inputs=[
                    'de_data_set@PKL',
                    'params:inputs',
                    'aircraft_categories_dict@PKL',
                    'params:pipeline_inspect_data_verbosity',
                    'params:data_inspector_verbosity',
                ],
                outputs='sklearn_pipeline_encode_steps',
                name='assemble_encode_steps',
            ),
            node(
                func=assemble_transform_steps,
                inputs=[
                    'params:inputs',
                    'params:pipeline_inspect_data_verbosity',
                    'params:data_inspector_verbosity',
                ],
                outputs='sklearn_pipeline_transform_steps',
                name='assemble_transform_steps',
            ),
            node(
                func=train_models,
                inputs=[
                    'de_data_set@PKL',
                    'sklearn_pipeline_impute_steps',
                    'sklearn_pipeline_encode_steps',
                    'sklearn_pipeline_transform_steps',
                    'params:models',
                    'params:inputs',
                    'params:inputs_train_only',
                    'params:target',
                    'params:globals',
                    'params:mlflow',
                    'mlflow_experiment_id',
                    'params:pipeline_inspect_data_verbosity',
                    'params:data_inspector_verbosity',
                ],
                outputs=[
                    'model_pipelines',
                    'model_mlflow_run_ids',
                ],
                name='train_models',
            ),
            node(
                func=predict_models,
                inputs=[
                    'model_pipelines',
                    'de_data_set@PKL',
                    'params:inputs',
                    'model_mlflow_run_ids',
                ],
                outputs='data_predicted',
                name='predict',
            ),
            node(
                func=report_models_performance_metrics,
                inputs=[
                    'data_predicted',
                    'params:target.name',
                    'params:models',
                    'params:metrics',
                    'model_mlflow_run_ids',
                ],
                outputs=None,
                name='report_models_performance_metrics',
            ),
            node(
                func=log_models_performance_artifacts,
                inputs=[
                    'data_predicted',
                    'params:target.name',
                    'params:models',
                    'model_mlflow_run_ids',
                ],
                outputs=None,
                name='log_models_performance_artifacts',
            ),
            node(
                func=log_models_sample_data,
                inputs=[
                    'de_data_set@PKL',
                    'params:models',
                    'params:inputs',
                    'params:target',
                    'model_mlflow_run_ids',
                ],
                outputs=None,
                name='log_models_sample_data',
            ),
            node(
                func=tag_with_arr_rwy_model_uri,
                inputs=[
                    'model_mlflow_run_ids',
                    'params:globals.arr_rwy_model_uri',
                ],
                outputs=None,
                name='tag_with_arr_rwy_model_uri',
            ),
        ],
        tags='ds',
    )

    return {
        'train_test': train_test_pipeline,
    }
