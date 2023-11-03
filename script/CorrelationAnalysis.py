import os
import pandas as pd
import numpy as np
import scipy.stats as st


def plot_heatmap_correlation(df, title_name, path_save_plot, fname, BIT=False):
    dict_color = {}
    dict_corr = {}
    PERC_REST = 0.9
    THRESHOLD = 0.5
    redundancy, complementary, heterogenous = [], [], []
    contador = 0
    lperc_corrx, lperc_corry = [], []
    for key in df.keys():
        # and (not 'Pu' in key) and (not 'P&' in key) and not ('P' == key[-1])
        if key != 'USER' and (not 'B' in key) and (not 'M&' in key) and (not 'ProbU_max' in key) and \
            (not 'E' in key) and (not 'K' in key) and (not 'Tau' in key) and (not 'ProbU' in key) and \
            (not 'ProbTl' in key) and (not 'Mu' in key) and not ('_group' in key) and (not '_group' in key) and \
            (not 'F' in key) and (not 'J' in key) and (not 'HcondO' in key) and (not 'HcondO_max' in key):
            ## (not 'MI' in key) and
            # print(key)
            contador += 1
            values = df[key].values.copy()
            values = values[~np.isnan(values)]
            size = values.size

            # print(key,': ',np.median(values).round(2))

            size_center = values[(values >= -THRESHOLD) & (values <= THRESHOLD)].size
            size_left = values[(values < -THRESHOLD)].size
            size_right = values[(values > THRESHOLD)].size
            # print(key,'%3.4f'%(size_left/size),'%3.4f'%(size_center/size),'%3.4f'%(size_right/size),end=', ')
            # Test of complementarity
            lperc_corrx.append((size_center / float(size)))
            lperc_corry.append(size_left / float(size) + size_right / float(size))
            if size_center / float(size) >= PERC_REST:
                # print(key,'\tcomplementary:',end=' ')
                # print('%.2f,%.2f'%((size_center/float(size)),size_left/float(size) + size_right/float(size)))
                percentage = 1 - size_center / float(size)
                dict_color[key] = percentage * 100  # 5#100#75
                dict_corr[key] = percentage
                # 5# 5 light red
                complementary.append(key)
            # Test of redundancy
            elif size_left / float(size) + size_right / float(size) >= PERC_REST:
                # print(key,'\tredundancy:',end=' ')
                # print('%.2f,%.2f'%((size_center/float(size)),size_left/float(size) + size_right/float(size)))
                percentage = size_left / float(size) + size_right / float(size)
                dict_color[key] = percentage * 100  # 100#0#25
                dict_corr[key] = percentage
                redundancy.append(key)
            else:
                # Heterogeneity
                if size_left / float(size) + size_right / float(size) >= 0.5:
                    # print(key,'\tredundancy h.:',end=' ')
                    # print('%.2f,%.2f'%((size_center/float(size)),size_left/float(size) + size_right/float(size)))
                    percentage = size_left / float(size) + size_right / float(size)
                    dict_color[key] = percentage * 100  # 75#25#0
                    dict_corr[key] = percentage
                    heterogenous.append(key)
                elif size_center / float(size) > 0.5:
                    # print(key,'\tcomplementary h.:',end=' ')
                    # print('%.2f,%.2f'%((size_center/float(size)),size_left/float(size) + size_right/float(size)))
                    percentage = 1 - size_center / float(size)
                    dict_color[key] = percentage * 100  # 20#75#100
                    heterogenous.append(key)
                    dict_corr[key] = percentage
                # print(size_left/float(size),size_right/float(size),size_center/float(size))

    # 'F','J' are recency category's media and user
    columns = ['D', 'C', 'G', 'Gu', 'H', 'P', 'Pu', 'MI', 'MI_max', 'MI_super', 'MI_super_max']
    ## ['D','C','G','Gu','H','P','Pu','ProbU','ProbU_max','HcondO','HcondO_max'
    dict_col = {}
    for i in range(len(columns)):
        dict_col[columns[i]] = i
    data = np.eye(len(columns), len(columns))
    data_corr = np.eye(len(columns), len(columns))
    # print(contador,dict_color.keys())
    # return
    for key in dict_color.keys():
        cols = key.split('&')
        # cols[0] == 'Pu' or cols[1] == 'Pu' or
        # or cols[0] == 'P' or cols[1] == 'P'
        if cols[0] == 'I' or cols[1] == 'I' or cols[0] == 'O' or cols[1] == 'O' or \
            cols[0] == 'Ou' or cols[1] == 'Ou' or 'M' == cols[1]:  ## 'M' in cols[0] or 'M' in cols[1] or
            continue
        # print(cols)
        data[dict_col[cols[0]], dict_col[cols[1]]] = dict_color[key]
        data[dict_col[cols[1]], dict_col[cols[0]]] = dict_color[key]
        # print(data[ dict_col[cols[0]], dict_col[cols[1]] ])
        # print(data[ dict_col[cols[1]], dict_col[cols[0]] ])

        data_corr[dict_col[cols[0]], dict_col[cols[1]]] = dict_corr[key]
        data_corr[dict_col[cols[1]], dict_col[cols[0]]] = dict_corr[key]
        # print(data_corr[ dict_col[cols[0]], dict_col[cols[1]] ])
        # print(data_corr[ dict_col[cols[1]], dict_col[cols[0]] ])

    # print(data)

    columns_raw = [r'Cat. Novelty', \
                   r'User Novelty', \
                   r'Cat. Uncertainty', \
                   r'User Uncertainty', \
                   r'Cat. Complexity',
                   r'Cat. Conflict',
                   r'User Conflict',
                   r'Avg. Direct Inf.',
                   r'Max. Direct Inf.',
                   r'Avg. Indirect Inf.',
                   r'Max. Indirect Inf.']

    columns_bit = columns_raw
    for row in range(data.shape[0]):
        data[row, row] *= 100

    if BIT == True:
        dheatmap = pd.DataFrame(data=data, columns=columns_bit)
        dheatmap.index = dheatmap.columns
    else:
        dheatmap = pd.DataFrame(data=data, columns=columns_raw)
        dheatmap.index = dheatmap.columns

    # print(dheatmap)

    import seaborn as sns
    import matplotlib.pyplot as plt

    # Generate a mask for the upper triangle
    mask = np.zeros_like(dheatmap.values, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots()
    mask_label_data = dheatmap.values
    """
    mask_label=[]
    for i in range(mask_label_data.shape[0]):
        for j in range(mask_label_data.shape[0]):
            if i > j:
                #print(mask_label_data[i,j])
                mask_label.append(mask_label_data[i,j])
    #print(mask_label_data)
    mask_label=np.array(mask_label).ravel()
    mask_label=mask_label >= 75
    """
    mask_label = []
    for i in range(mask_label_data.shape[0]):
        label = ''
        for j in range(mask_label_data.shape[0]):
            if mask_label_data[i, j] == 25:
                label += 'c,'
            if mask_label_data[i, j] == 50:
                label += 'hc,'
            if mask_label_data[i, j] == 75:
                label += 'hr,'
            if mask_label_data[i, j] == 100:
                label += 'r,'
            if mask_label_data[i, j] == 1:
                label += '$,'
                # print(label[:-1].split(','))
        mask_label.append(label[:-1].split(','))

    mask_label = np.array(mask_label)
    # print(mask_label)
    mask = np.zeros_like(dheatmap.values, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)  # sns.diverging_palette(100, 2, n=3,as_cmap=True)
    # sns.color_palette("Blues")
    # sns.diverging_palette(220, 250, n=4,sep=25,as_cmap=True)
    cbar_kws = {'ticks': np.arange(0, 101, 25), "drawedges": False,
                'shrink': 0.99}  # 'label':'Intensity of relation among variables'}
    annot_kws = {"size": 16, 'weight': 'normal', 'fontstyle': 'normal', 'color': 'black'}

    sns.set(style="white", font_scale=1.5)
    sns.set_style('whitegrid', {'font.family': 'sans-serif', 'font.serif': 'Times New Roman'})

    plt.subplots(figsize=(14, 12))
    plt.close(fig)

    data_annot = data_corr * 100
    data_annot = data_annot.round(0)
    data_annot = data_annot.astype(int)
    data_annot = data_annot.astype('<U32')
    data_annot = np.char.add(data_annot, '%')
    # print(data_annot)

    sns.heatmap(dheatmap, linewidths=2.0, mask=mask, cmap=cmap, square=True, cbar=True, center=0, \
                annot=data_corr, annot_kws=annot_kws, cbar_kws=cbar_kws, fmt='.0%')
    plt.savefig(path_save_plot + fname + '.png', bbox_inches='tight', dpi=300)
    # vmin=-1,vmax=100
    """
    sns.heatmap(dheatmap,linewidths=0.5, mask=mask,fmt='.0f',\
                cmap=cmap,square=True,cbar=False,vmin=0,vmax=100,center=-10,\
                annot=data_corr*100,annot_kws=annot_kws,cbar_kws=cbar_kws)#fmt='.1f',cmap = 'inferno'
    """
    """
    for text, show_annot in zip(ax.texts, (element for element in mask_label)):
        text.set_visible(show_annot)    
    """

    plt.yticks(rotation=0)
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif', 'font.serif': ['Times']})
    # plt.title(title_name)
    # plt.show()

    plt.close()

    return redundancy, complementary, heterogenous, dheatmap, lperc_corrx, lperc_corry


def plot_cdf(df, path_save, fname, field):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.ecdfplot(data=df, x=field)

    plt.savefig(path_save + fname + '.png', bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    path_corr = './stimulus-bits/'
    path_save_plot = './correlation-analysis-results/'

    dspearman = pd.read_csv(path_corr + 'spearman-0.tsv', sep='\t')
    dpearson = pd.read_csv(path_corr + 'pearson-0.tsv', sep='\t')

    # dspearman.dropna(inplace=True)
    # dpearson.dropna(inplace=True)
    dspearman.fillna(0, inplace=True)
    dpearson.fillna(0, inplace=True)

    dspearman.columns = dspearman.columns.str.replace('_avg', '')
    dpearson.columns = dpearson.columns.str.replace('_avg', '')
    dspearman.columns = dspearman.columns.str.replace('_sum', '')
    dpearson.columns = dpearson.columns.str.replace('_sum', '')

    rspear_bit, cspear_bit, hspear_bit, dheatmap, lperc_corrx, lperc_corry = plot_heatmap_correlation(dspearman,
                                                                                                      "Spearman Bit-Data",
                                                                                                      path_save_plot,
                                                                                                      'spearman-plt',
                                                                                                      BIT=True)

    rspear_bit, cspear_bit, hspear_bit, dheatmap, lperc_corrx, lperc_corry = plot_heatmap_correlation(dpearson,
                                                                                                      "Pearson Bit-Data",
                                                                                                      path_save_plot,
                                                                                                      'pearson-plt',
                                                                                                      BIT=True)

    plot_cdf(dspearman, path_save_plot, 'direct_inf_max&avg_spearman', 'MI_max&MI')
    plot_cdf(dspearman, path_save_plot, 'inf_ind_max&avg_spearman', 'MI_super_max&MI_super')
    plot_cdf(dspearman, path_save_plot, 'cat_cnf&cpx_spearman', 'P&H')
    plot_cdf(dspearman, path_save_plot, 'user_cnf&uncertainty_spearman', 'Pu&Gu')

    plot_cdf(dpearson, path_save_plot, 'direct_inf_max&avg_pearson', 'MI_max&MI')
    plot_cdf(dpearson, path_save_plot, 'inf_ind_max&avg_pearson', 'MI_super_max&MI_super')
    plot_cdf(dpearson, path_save_plot, 'cat_cnf&cpx_pearson', 'P&H')
    plot_cdf(dpearson, path_save_plot, 'user_cnf&uncertainty_pearson', 'Pu&Gu')
