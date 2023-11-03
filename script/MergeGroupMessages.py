import os
import pandas as pd
from tqdm import tqdm


def group_data_merge(df, fpath, columns):
    file_df = pd.read_csv(fpath, sep='\t')

    df_out = pd.concat([df, file_df.loc[:, columns]])
    return df_out


if __name__ == "__main__":
    data_path = './stimulus-bits/users'
    save_path = './cluster-results/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    files = os.listdir(data_path)

    get_columns_df = pd.read_csv(data_path + '/' + files[0], sep='\t')
    df_columns = get_columns_df.columns
    del get_columns_df

    data_df = pd.DataFrame(columns=df_columns)

    print('Start merging groups')
    for file in tqdm(files):
        data_df = group_data_merge(data_df, data_path + '/' + file, df_columns)

    data_df.to_csv(save_path + 'merged_data.tsv', sep='\t')
