import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from multiprocessing import Process


def plot_interactive_scatter(k):
    plots_df = pd.read_csv('./cluster-results/users-profiles/clusters-3-to-6/clusters-k-' + str(k) + '.csv',
                           sep='\t')

    sns.set_style("darkgrid")
    fig = plt.figure(figsize=(7, 6))
    plot_axes = Axes3D(fig)
    plot = plot_axes.scatter(plots_df['0-0'], plots_df['0-1'], plots_df['0-2'],
                             c=list(map(lambda x: int(x), plots_df['Labels'])), cmap="Set1")

    legend1 = plot_axes.legend(*[plot.legend_elements()[0], [('Cluster %d' % i) for i in range(k)]],
                               title="Legend", loc='upper left')
    plot_axes.add_artist(legend1)

    plt.title('Clusterização para K=%d' % k)
    plot_axes.set_xlabel('PC 1')
    plot_axes.set_ylabel('PC 2')
    plot_axes.set_zlabel('PC 3')

    # for angle in range(0, 360):
    #     plot_axes.view_init(angle, 30)
    #     plt.draw()
    #     plt.pause(.001)

    plt.show()


if __name__ == '__main__':
    max_k = 6
    min_k = 3

    N_PROC = max_k - min_k
    block = []

    for i in range(N_PROC):
        blk = Process(target=plot_interactive_scatter, args=([i + 3]))
        blk.start()
        print('K =', i + 3, 'on process', blk.pid, "starting...")
        block.append(blk)

    for blk in block:
        print('%d,%s' % (blk.pid, " waiting..."))
        blk.join()

