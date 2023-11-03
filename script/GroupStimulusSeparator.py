import os
import pandas as pd
from tqdm import tqdm


if __name__ == '__main__':
    df = pd.read_csv('./cluster-results/merged_data.tsv', sep='\t')
    groups = list(map(lambda x: str(x), (pd.read_csv('./dataset_gname_raw_info.csv'))['gname'].values))

    save_dir = './groups/groups-stimulus'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    df['group'] = df.group.astype('string')

    num_saves = 0
    for group in tqdm(groups):
        group_data = df[df.group.str.contains(group)]
        if group_data.shape[0] != 0:
            num_saves = num_saves + 1
            group_data.to_csv(save_dir + '/' + group + '.tsv', sep='\t')

    print("Finishing group divider.")
    print("Saved %d groups" % num_saves)
