import pandas as pd
import os



TABLE_DIR = 'data/lookup_table'


def create_table(alg):
    if alg == 'raw':
        columns = ['id', 'T']
    elif alg == 'gaussian':
        columns = ['id', 'rep', 'T', 'I', 'eps', 'delta']
    elif alg == 'dft':
        columns = ['id', 'rep', 'T', 'I', 'eps', 'delta', 'k']
    elif alg == 'ss':
        columns = ['id', 'rep', 'T', 'I', 'eps', 'delta', 'k', 'interpolate_kind']
    elif alg == 'ssf':
        columns = ['id', 'rep', 'T', 'I', 'eps', 'delta', 'k', 'interpolate_kind', 'std']

    df = pd.DataFrame(columns=columns)
    return df


def read_table(alg):
    file_path = os.path.join(TABLE_DIR, f'{alg}.pkl')
    if os.path.exists(file_path):
        df = pd.read_pickle(file_path)
    else:
        df = create_table(alg)
    return df


def add_row(alg, row):
    file_path = os.path.join(TABLE_DIR, f'{alg}.pkl')
    df = read_table(alg)
    df = df.append(row, ignore_index=True)
    df.to_pickle(file_path)


def search_id(alg, param):
    df = read_table(alg)
    target_rows = df.loc[(df[list(param)] == pd.Series(param)).all(axis=1)]
    if len(target_rows) != 1:
        print('multiple rows matching the param')
    id_str = target_rows.iloc[0]['id']
    return id_str

'''
When manually add columns for new parameters:
`df[param_name] = param_value`.
When manually remove rows:
`df.drop(labels=indices, inplace=True)`
'''
