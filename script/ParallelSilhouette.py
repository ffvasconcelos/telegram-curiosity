import os
import sys

import numpy as np
import math
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score
from sklearn.cluster import MiniBatchKMeans

from multiprocessing import Process, Array
from multiprocessing.pool import ThreadPool as Pool

import random

"""
sample = 151318
max_k = 18
min_k = 3
pool_number = 6
silhouette_array = 10
"""

sample = 100
max_k = 18
min_k = 3
pool_number = 1
silhouette_array = 10


def get_silhouette_score(values, k, index, array):
    print("Getting silhouette %d for k = %d" % (index, k))
    random_seed = random.randint(min_k, max_k)
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=random_seed).fit(values)
    labels = kmeans.labels_
    array[index] = silhouette_score(values, labels, sample_size=sample)
    print("Finishing silhouette %d for k = %d" % (index, k))


def get_silhouette_mean(k, values):
    print("Starting silhouette scores acquisition for k=%d" % k)
    scores = Array('d', silhouette_array)
    block = []

    for i in range(silhouette_array):
        blk = Process(target=get_silhouette_score, args=(values, k, i, scores))
        blk.start()
        block.append(blk)

    for blk in block:
        blk.join()

    print("Finishing silhouette score acquisition")
    return scores[:]


def generate_silhouette_scores(values):
    pool = Pool(processes=pool_number)
    print("Starting process pool")
    results = pool.starmap_async(get_silhouette_mean, [(k, values) for k in range(min_k, max_k + 1)])
    pool.close()
    pool.join()

    return results.get()


if __name__ == "__main__":
    if not os.path.exists('./cluster-results/'):
        print('No path detected aborting')
        sys.exit()

    df = pd.read_csv('./cluster-results/transformed_df.tsv', sep='\t')

    silhouette = generate_silhouette_scores(df[['MI_max', 'MI_super_max']].values)

    silhouette_means = list(map(np.mean, silhouette))
    silhouette_std = list(map(np.std, silhouette))
    confidence_coefficient = 1 - 0.05

    confidence_interval = list(map(lambda mean: (mean - (1.960 * confidence_coefficient / math.sqrt(sample)),
                                                 mean + (1.960 * confidence_coefficient / math.sqrt(sample))),
                                   silhouette_means))

    sns.lineplot(x=range(min_k, max_k + 1), y=silhouette_means, markers=True)
    sns.lineplot(x=range(min_k, max_k + 1), y=[x[0] for x in confidence_interval], color='purple')
    sns.lineplot(x=range(min_k, max_k + 1), y=[x[1] for x in confidence_interval], color='purple')

    plt.savefig('./cluster-results/silhouette-index.pdf', bbox_inches='tight', dpi=300)

