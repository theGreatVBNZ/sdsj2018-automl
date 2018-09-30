import argparse
import os
import pickle
import time
from functools import partial

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from utils import transform_data, make_predictions, root_mean_squared_error, pprint


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
    if is_big:
        pipeline, df_x = transform_data(df_x, target)
    else:
        pipeline, df_x = transform_data(df_x, target)

    if args.mode == 'regression':
        model = Ridge(alpha=30, fit_intercept=True)
        model.fit(df_x, target)
        pipeline.append(partial(make_predictions, model=model))

        rmse = root_mean_squared_error(model.predict(df_x), target)
        print(f'RMSE: {rmse}')

        pipeline_path = os.path.join(args.model_dir, 'pipeline.pkl')
        with open(pipeline_path, 'wb') as pickle_file:
            pickle.dump(pipeline, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        model = GradientBoostingClassifier(n_estimators=100)
        model.fit(df_x, target)
        pipeline.append(partial(make_predictions, model=model, proba=True))

        roc_auc = roc_auc_score(target, model.predict_proba(df_x)[:, 1])
        print(f'ROC-AUC: {roc_auc}')

        pipeline_path = os.path.join(args.model_dir, 'pipeline.pkl')
        with open(pipeline_path, 'wb') as pickle_file:
            pickle.dump(pipeline, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed_time = time.time() - start_time
    print(f'Train time: {elapsed_time}')


if __name__ == '__main__':
    main()
