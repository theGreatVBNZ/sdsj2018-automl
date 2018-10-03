import os
import datetime
from functools import lru_cache, partial
from itertools import product
from IPython.display import display

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# @lru_cache()
def load(task, table, **kwargs):
    """
    :param task: Номер задачи
    :param table: Имя таблицы
    :return: X, y
    """
    task_postfix = {i: 'r' if i < 4 else 'c' for i in range(1, 9)}
    data_dir = 'data'
    tables_dir = f'check_{task}_{task_postfix[task]}'
    data_path = os.path.join('..', data_dir, tables_dir)

    if '.csv' in table:
        table_name = table
    else:
        table_name = f'{table}.csv'

    if table_name not in ('train.csv', 'test.csv', 'test-target.csv'):
        raise FileNotFoundError(table_name)

    table_path = os.path.join(data_path, table_name)
    df = pd.read_csv(table_path, **kwargs)
    target = None
    if 'train' in table_name:
        target = df['target']
        df.drop('target', axis=1, inplace=True)
    elif 'test-target.csv' in table_name:
        target = df.set_index('line_id')
        df = None
    elif 'test.csv' in table_name:
        target = None
    return df, target


def pprint(*args):
    """
    DEBUG PRINT
    """
    print('------------------------------------')
    print(' '.join([str(arg) if not isinstance(arg, float) else str(round(arg, 2)) for arg in args]))
    print('----------***************-----------')


def parse_dt(x):
    if not isinstance(x, str):
        return None
    elif len(x) == len('2010-01-01'):
        return datetime.datetime.strptime(x, '%Y-%m-%d')
    elif len(x) == len('2010-01-01 10:10:10'):
        return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    else:
        return None
    

def select_datetime_columns(df):
    datetime_columns = [
        col_name
        for col_name in df.columns
        if col_name.startswith('datetime')
    ]
    return datetime_columns


def transform_datetime_features(df_x, datetime_columns):
    df_datetime = pd.DataFrame()
    for col_name in datetime_columns:
        col = df_x[col_name].apply(lambda x: parse_dt(x))
        df_x[col_name] = col # Inplace Transform
        df_datetime[f'dt_number_weekday_{col_name}'] = col.apply(lambda x: x.weekday())
        df_datetime[f'dt_number_month_{col_name}'] = col.apply(lambda x: x.month)
        df_datetime[f'dt_number_day_{col_name}'] = col.apply(lambda x: x.day)
        df_datetime[f'dt_number_hour_{col_name}'] = col.apply(lambda x: x.hour)
        df_datetime[f'dt_number_hour_of_week_{col_name}'] = col.apply(lambda x: x.hour + x.weekday() * 24)
        df_datetime[f'dt_number_minute_of_day_{col_name}'] = col.apply(lambda x: x.minute + x.hour * 60)
    df_x = pd.concat([df_x, df_datetime], axis=1)
    return df_x


def constant_features(df_x):
    constant_columns = [
        col_name
        for col_name in df_x.columns
        if df_x[col_name].nunique() == 1
    ]
    return constant_columns


def drop_columns(df_x, cols):
    return df_x.drop(cols, axis=1)


def select_for_encoding(df_x, max_unique=20):
    categorical = {}
    for col_name in list(df_x.columns):
        if col_name.startswith('dt'):
            continue

        col_unique_values = df_x[col_name].unique()
        if 2 < len(col_unique_values) <= max_unique:
            categorical[col_name] = col_unique_values
    return categorical


def one_hot_encoding(df_x, categorical):
    df_dummies = pd.DataFrame()
    for col_name, unique_values in categorical.items():
        for unique_value in unique_values:
            df_dummies[f'onehot_{col_name}={unique_value}'] = (df_x[col_name] == unique_value).astype(int)
    return pd.concat([df_x, df_dummies], axis=1)


def find_missings(df_x):
    columns_with_missings = df_x.columns[df_x.isna().any()].tolist()
    return columns_with_missings


def fill_missings(df_x, columns):
    df_x = df_x.copy()
    for col_name in columns:
        df_x[col_name].fillna(-1, inplace=True)
    return df_x


def select_numeric_columns(df_x):
    numeric_columns = [
        col_name
        for col_name in df_x.columns
        if col_name.startswith('number') or col_name.startswith('onehot') or col_name.startswith('dt')
    ]
    return numeric_columns


def keep_columns(df_x, columns):
    return df_x[columns].copy()


def create_scaler(df_x):
    scaler = StandardScaler()
    scaler.fit(df_x)
    return scaler


def scale(df_x, scaler):
    return pd.DataFrame(data=scaler.transform(df_x), columns=df_x.columns)


def make_predictions(df_transformed, model, proba=False):
    if proba:
        predictions = model.predict_proba(df_transformed)[:, 1]
    else:
        predictions = model.predict(df_transformed)
    return predictions


def transform_data(df_x, target):
    # TODO: inplace functions
    pipeline = []

    datetime_columns = select_datetime_columns(df_x)
    df_x = transform_datetime_features(df_x, datetime_columns)
    pipeline.append(partial(transform_datetime_features, datetime_columns=datetime_columns))
    
    constant_columns = constant_features(df_x)
    df_x = drop_columns(df_x, cols=constant_columns)
    pipeline.append(partial(drop_columns, cols=constant_columns))

    categorical = select_for_encoding(df_x, max_unique=20)
    df_x = one_hot_encoding(df_x, categorical=categorical)
    pipeline.append(partial(one_hot_encoding, categorical=categorical))

    columns_with_missings = find_missings(df_x)
    df_x = fill_missings(df_x, columns=columns_with_missings)
    pipeline.append(partial(fill_missings, columns=columns_with_missings))

    numeric_columns = select_numeric_columns(df_x)
    df_x = keep_columns(df_x, columns=numeric_columns)
    pipeline.append(partial(keep_columns, columns=numeric_columns))

    scaler = create_scaler(df_x)
    df_x = scale(df_x, scaler)
    pipeline.append(partial(scale, scaler=scaler))

    df_transformed = df_x
    return pipeline, df_transformed


def predict(df_x, pipeline):
    result = df_x
    for transform in pipeline:
        result = transform(result)
    return result


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def models_factory(base_model, **params):
    products = list(product(*params.values()))
    configs = pd.DataFrame(data=products, columns=list(params.keys())).to_dict('records')
    models = [base_model().set_params(**config) for config in configs]
    for model in models:
        yield model


def assess(model):
    print('----------------------------\n', model, '\n')
    train_err, valid_err, test_err = [], [], []
    for i in range(10):
        df_x, target = load(1, 'train')
        _, y_test = load(1, 'test-target')
        x_test, _ = load(1, 'test')

        x_train, x_valid, y_train, y_valid = train_test_split(df_x, target, test_size=.2, random_state=i * 1234)

        pipeline, x_train = transform_data(x_train, y_train)

        model.fit(x_train, y_train)
        pipeline.append(partial(make_predictions, model=model))

        train_predictions = make_predictions(x_train, model)
        train_rmse = root_mean_squared_error(y_train, train_predictions)
        train_err.append(train_rmse)

        valid_predictions = predict(x_valid, pipeline)
        valid_rmse = root_mean_squared_error(y_valid, valid_predictions)
        valid_err.append(valid_rmse)

        test_predictions = predict(x_test, pipeline)
        test_rmse = root_mean_squared_error(y_test, test_predictions)
        test_err.append(test_rmse)

    #     print('----------------------------')
    #     print(f'Train RMSE: {train_rmse}, \nValidation RMSE: {valid_rmse}, \nTest RMSE: {test_rmse}')
    #     print('----------------------------')
    df_err = pd.DataFrame(dict(train=train_err, valid=valid_err, test=test_err))
    display(df_err.mean())
    df_err.plot(kind='bar')
    plt.show()
    print('----------------------------')
    return df_err
