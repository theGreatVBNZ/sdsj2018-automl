import os
# from functools import lru_cache

from pandas import read_csv


# @lru_cache()
def load(task, table, **kwargs):
    """
    :param task: Номер задачи
    :param table: Имя таблицы
    :return: X, y
    """
    task_postfix = {i: 'r' if i < 3 else 'c' for i in range(1, 9)}
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
    df = read_csv(table_path, **kwargs)
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
