import argparse
import os
import pickle
import time
from functools import partial

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score

from utils import (
    group_columns_by_type,
    create_datetime_transformer,
    create_lag_transformer,
    create_feature_generator,
    create_pipeline,
    root_mean_squared_error,
    pprint
)


TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))
BIG_DATASET_SIZE = 500 * 1024 * 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', type=argparse.FileType('r'), required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--mode', choices=['classification', 'regression'], required=True)
    args = parser.parse_args()

    start_time = time.time()

    df_x = pd.read_csv(args.train_csv)
    target = df_x['target']
    df_x.drop('target', axis=1, inplace=True)
    is_big = df_x.memory_usage().sum() > BIG_DATASET_SIZE

    print(f'Dataset read, shape {df_x.shape}, \nBig: {is_big}')

    # TODO: Создать pipeline для больших датасетов
    datetime_columns, numerical_columns, categorical_columns, idx_columns, single_value_columns = group_columns_by_type(df_x, 10)
    datetime_transformer = create_datetime_transformer(datetime_columns)
    lag_transformer = create_lag_transformer(datetime_columns, numerical_columns)
    feature_generator = create_feature_generator(numerical_columns, datetime_transformer, lag_transformer)
    pipeline = create_pipeline(feature_generator)
    print('Data Processing Pipeline:', pipeline)

    x = pipeline.fit_transform(df_x)
    print('Processing is over ...')

    pipeline_path = os.path.join(args.model_dir, 'pipeline.pkl')
    with open(pipeline_path, 'wb') as pickle_file:
        pickle.dump(pipeline, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    if args.mode == 'regression':
        model = RandomForestRegressor(bootstrap=True,
                max_depth=53, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=169, n_jobs=-1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)
        model.fit(x, target)
        rmse = root_mean_squared_error(model.predict(x), target)
        print(f'RMSE: {rmse}')
    else:
        model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=53, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=169, n_jobs=-1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)
        model.fit(x, target)
        roc_auc = roc_auc_score(target, model.predict_proba(x)[:, 1])
        print(f'ROC-AUC: {roc_auc}')

    model_path = os.path.join(args.model_dir, 'model.pkl')
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed_time = time.time() - start_time
    print(f'Train time: {elapsed_time}')


if __name__ == '__main__':
    main()
