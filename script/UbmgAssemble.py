import numpy as np
import pandas as pd


def assemble_matrix(data):
    matrix = np.zeros((3, 3))
    values = data.labels.values

    for i in range(0, len(values) - 1):
        matrix[values[i], values[i + 1]] += 1

    for i in range(3):
        matrix[i, :] /= (1 if matrix[i, :].sum() == 0 else matrix[i, :].sum())

    return matrix


def assemble_ubmg(users, data_path, save_path):
    for user in users:
        df = pd.read_csv(data_path + '/' + user + '.tsv', sep='\t')
        ubmg = assemble_matrix(df)
        np.savetxt(save_path + '/' + user + '.txt', ubmg, fmt='%f', delimiter=',')


def flatten_matrixes(users, data_path, save_path):
    for user in users:
        matrix = np.loadtxt(data_path + '/' + user + '.txt', dtype=float, delimiter=',')
        flattened_matrix = matrix.flatten()
        np.savetxt(save_path + '/' + user + '.txt', flattened_matrix, fmt='%f', delimiter=',')


def ubmg_df(users, data_path, save_path):
    users_df = pd.DataFrame(columns=['user', '0-0', '0-1', '0-2', '1-0', '1-1', '1-2', '2-0', '2-1', '2-2'])

    for user in users:
        data = np.loadtxt(data_path + '/' + user + '.txt', dtype=float, delimiter=',')
        users_df = pd.concat([users_df, pd.DataFrame({
            'user': [user],
            '0-0': [data[0]],
            '0-1': [data[1]],
            '0-2': [data[2]],
            '1-0': [data[3]],
            '1-1': [data[4]],
            '1-2': [data[5]],
            '2-0': [data[6]],
            '2-1': [data[7]],
            '2-2': [data[8]]
        })])

    users_df.to_csv(save_path + '/users-profile.tsv', sep='\t')


def centroids_to_matrix(data_path, save_dir, k):
    df = pd.read_csv(data_path, sep='\t')
    matrixes = [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))]

    for cluster in range(k):
        for i in range(3):
            for j in range(3):
                matrixes[cluster][i, j] = df[str(i) + '-' + str(j)][cluster]

    for cluster in range(k):
        for i in range(3):
            matrixes[cluster][i, :] /= (1 if matrixes[cluster][i, :].sum() == 0 else matrixes[cluster][i, :].sum())

    for cluster in range(k):
        print('Saving matrix', cluster)
        np.savetxt(save_dir + '/centroid-matrix-' + str(cluster) + '.txt', matrixes[cluster], fmt='%f', delimiter=',')



"""
if __name__ == "__main__":
    df = np.loadtxt('ubmg-data/flattened/9646704--1001494170309.txt', dtype=float, delimiter=',')
    assemble_matrix(df)
"""
