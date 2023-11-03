#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 16:56:55 2018

@author: alexandre
"""
import os
import numpy as np
import pandas as pd
from logger import Logger
from multiprocessing import Process
from tqdm import tqdm


def recording_files(list_dt, list_cols, STORE_PATH):
    for i in range(len(list_cols)):
        fname2 = 'pearson-' + str(i) + '.tsv'
        dt1 = pd.DataFrame(np.array(list_dt[i * 2]).T, columns=list_cols[i])
        fname3 = 'spearman-' + str(i) + '.tsv'
        dt2 = pd.DataFrame(np.array(list_dt[i * 2 + 1]).T, columns=list_cols[i])
        if not os.path.exists(STORE_PATH + '/' + fname2) and \
            not os.path.exists(STORE_PATH + '/' + fname3):  # FLAG==False:
            dt1.to_csv(STORE_PATH + '/' + fname2, sep='\t', header=True, float_format='%.2f', index=False)
            dt2.to_csv(STORE_PATH + '/' + fname3, sep='\t', header=True, float_format='%.2f', index=False)
        else:
            with open(STORE_PATH + '/' + fname2, 'a') as f:
                dt1.to_csv(f, sep='\t', header=False, float_format='%.2f', index=False)

            with open(STORE_PATH + '/' + fname3, 'a') as f:
                dt2.to_csv(f, sep='\t', header=False, float_format='%.2f', index=False)


def entropy_compute_time(arr, delta):
    h = 0.
    if type(arr) == float:
        prob = arr
        if prob == 0.:
            # prob=delta ## the entropy of zero probability is zero by convention
            h = 0.  # np.log2(prob)
        else:
            prob = prob / delta
            h = -prob * np.log2(prob)
    else:
        prob = arr[arr != 0]
        if prob.size > 0:
            prob = prob / delta
            h = -prob * np.log2(prob)
        else:
            # prob=delta
            h = np.array([0.])  # np.log2(prob) ## the entropy of zero probability is zero by convention
        h = h.sum()
    return h


def entropy_compute(arr1):
    h = 0.
    if type(arr1) == float:
        prob = arr1
        if prob == 0.:
            h = 0.
        else:
            h = -prob * np.log2(prob)
    else:
        prob = arr1[arr1 != 0]
        if prob.size == 0:
            h = np.array([0.])
        else:
            h = -prob * np.log2(prob)
        h = h.sum()
    return h


def mutual_info_compute(arr1, arr2, arr3):
    den = (arr2 * arr3)
    if den[den != 0].size > 0:
        mi = np.max(arr1[den != 0] / den[den != 0])
    else:
        mi = 0.
    return mi


"""
Remove the Zero values if there is at least one value different of zero
"""


def remove_zero(arr):
    idx = np.nonzero(arr)
    arr = arr[idx]
    if arr.size == 0:
        arr = np.array([0])
    return arr


def divide_arrays_by_cell(arr1, arr2):
    i = np.nonzero(arr2)[0]
    # print(arr1,arr1.dtype,arr2,arr2.dtype)
    if i.size == 0:
        result = np.zeros(arr1.shape, dtype=float)
    else:
        result = arr1[i] / arr2[i]
    # print(result,result.dtype)
    result = result[np.nonzero(result)]
    if result.size == 0:
        result = np.array([0])
    return result


def divide_arrays_by_cell_scalar(arr1, arr2):
    i = np.nonzero(arr2)[0]  # get the list of index of non zero values from tuple
    # print(i,j,arr1,arr1.dtype,arr2,arr2.dtype)
    if i.size == 0:
        result = np.zeros(arr1.shape, dtype=float)
    else:
        result = arr1 / arr2[i]
    # print(result,result.dtype)
    result = result[np.nonzero(result)]
    if result.size == 0:
        result = np.array([0])
    return result


def logarithm_weighted_array(arr1, arr2):
    arr1[arr1 == 0] = 1.
    arr1 = np.log2(arr1)
    result = arr2 * arr1
    result[result < 0] = 0.
    return result


def logarithm_array(arr):
    arr[arr == 0] = 1.
    arr = np.log2(arr)
    arr[arr < 0] = 0.
    return arr


def maximum_array(arr):
    arr[arr < 0] = 0.
    return arr


def largest_count_to_surprisal_compute(arr1, arr2):
    # print(arr1,arr2)
    i = np.argmax(arr1)
    if arr2.size > 1 and arr2[i] != 0:
        p = arr1[i] / arr2[i]
    elif arr2[0] != 0:
        p = arr1[i] / arr2[0]
    else:
        p = np.array([0.])
    return p


def tau_largest_count_to_surprisal_compute(arr, tau_arr):
    i = np.argmax(arr)
    t = tau_arr[i]
    return t


def working_process_bits(logger, slices, path, STORE_PATH, version):
    count = 0
    if not os.path.exists(STORE_PATH + '/stats-' + str(version) + '/'):
        os.makedirs(STORE_PATH + '/stats-' + str(version) + '/')
    if not os.path.exists(STORE_PATH + '/users/'):
        os.makedirs(STORE_PATH + '/users/')
    ldata, ldata2, ldata10, ldata11, ldata8, ldata9, ldatagroup = [], [], [], [], [], [], []
    cols1 = list('BCDFGHIM')  # list('BCDEFGHI')
    cols1 += ['J_avg', 'O_sum', 'P_avg', 'Gu', 'Mu', 'Ou_sum', 'Pu_avg', \
              'ProbU_avg', 'MI_super_avg', 'MI_avg', 'Tau_avg', \
              'ProbU_max', 'MI_super_max', 'MI_max', 'Tau_max',
              'HcondO_avg', 'HcondO_max']  # ,\
    # 'HCONDO_group','MI_group']

    cols_corrs = ['USER']
    for row in range(1, len(cols1)):
        for col in range(0, row):
            cols_corrs.append('%s&%s' % (cols1[row], cols1[col]))

    col_st = ['user', 'avg', 'std', 'min', '10%', '50%', '90%', 'max']
    dict_dtype = {}
    for key in col_st:
        dict_dtype[key] = float
    for i in range(len(cols1)):
        ldata.append([])
        ldatagroup.append([])
    for i in range(len(cols1)):
        ldata2.append([])
    for i in range(len(cols_corrs)):
        ldata10.append([])
        ldata11.append([])
    for i in range(len(cols_corrs)):
        ldata8.append([])
        ldata9.append([])
    # prvalues=[]
    pcvalues = []
    tauvalues = []
    TIME_WINDOW = 1800
    # Dt=30*60
    count = records = 0
    RECORDS_PERIOD = 1
    for fname in slices:
        # try:

        print('\n\t', fname, '\n')
        count += 1
        cols = ['user', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'Gu', 'H', 'I', 'L', 'M', 'N', \
                'J', 'O', 'P', 'Mu', 'Ou', 'Pu', 'Nu', 'ProbTl', 'PindK', 'PindU', \
                'ProbU', 'Tau', 'PO', 'Hd_partial', \
                'HcondO', 'HCONDO', 'HD', 'Us', 'TotalU', 'total', 'timestamp']

        if os.path.exists(path + "/stimulus-filter/" + str(fname) + '.txt'):
            dframe = pd.read_csv(path + '/stimulus-filter/' + \
                                 str(fname) + '.txt', sep='\t', names=cols, comment='#', \
                                 keep_default_na=True)
            dframe.fillna(0., inplace=True)

            users_idx = dframe.user.unique()

            """ 
            ------------------------------------------------------------------------------------
            """
            if dframe[dframe.user.isin(users_idx)].shape[0] >= 30:
                if dframe.J.dtype != np.int64:
                    dframe.J = dframe.J.str.split(',').apply(np.asarray, args=(int,))
                if dframe.P.dtype != np.float64:
                    dframe.P = dframe.P.str.split(',').apply(np.asarray, args=(float,))
                if dframe.O.dtype != np.int64:
                    dframe.O = dframe.O.str.split(',').apply(np.asarray, args=(int,))
                if dframe.Pu.dtype != np.float64:
                    dframe.Pu = dframe.Pu.str.split(',').apply(np.asarray, args=(float,))
                if dframe.Ou.dtype != np.int64:
                    dframe.Ou = dframe.Ou.str.split(',').apply(np.asarray, args=(int,))
                if dframe.N.dtype != np.int64:
                    dframe.N = dframe.N.str.split(',').apply(np.asarray, args=(int,))

                # dframe.ProbTl = dframe.ProbTl.astype(str)
                # dframe.ProbTl = dframe.ProbTl.str.split(',').apply(np.asarray,args=(float,))
                dframe.ProbU = dframe.ProbU.astype(str)
                dframe.ProbU = dframe.ProbU.str.split(',').apply(np.asarray, args=(float,))
                dframe.Tau = dframe.Tau.astype(str)
                dframe.Tau = dframe.Tau.str.split(',').apply(np.asarray, args=(float,))
                dframe.PindU = dframe.PindU.astype(str)
                dframe.PindU = dframe.PindU.str.split(',').apply(np.asarray, args=(float,))
                dframe.PindK = dframe.PindK.astype(str)
                dframe.PindK = dframe.PindK.str.split(',').apply(np.asarray, args=(float,))
                dframe.total = dframe.total.astype(str)
                dframe.total = dframe.total.str.split(',').apply(np.asarray, args=(float,))
                dframe.ProbTl = dframe.ProbTl.astype(str)
                dframe.ProbTl = dframe.ProbTl.str.split(',').apply(np.asarray, args=(float,))

                dframe.HcondO = dframe.HcondO.astype(str)
                dframe.HcondO = dframe.HcondO.str.split(',').apply(np.asarray, args=(float,))

                dframe.PO = dframe.PO.astype(str)
                dframe.PO = dframe.PO.str.split(',').apply(np.asarray, args=(float,))

                dframe.Hd_partial = dframe.Hd_partial.astype(str)
                dframe.Hd_partial = dframe.Hd_partial.str.split(',').apply(np.asarray, args=(float,))

                dframe['PoHcondO'] = dframe.PO * dframe.HcondO
                dframe['MI_super'] = dframe['Hd_partial'] - dframe['PoHcondO']
                # get the maximum between 0 and every value of MI_super array
                dframe['MI_super'] = np.vectorize(maximum_array, otypes=[object])(dframe['MI_super'])
                dframe['MI_super_avg'] = dframe['MI_super'].apply(np.mean)
                dframe['MI_super_max'] = dframe['MI_super'].apply(np.max)

                # take the probabilities with the largest couting of numerator
                # dframe['ProbU_count']=np.vectorize(largest_count_to_surprisal_compute)(dframe['ProbU'],dframe['PindK'])
                # dframe['ProbTl_count']=np.vectorize(largest_count_to_surprisal_compute)(dframe['ProbTl'],dframe['total'])
                # dframe['Tau_count']=np.vectorize(tau_largest_count_to_surprisal_compute)(dframe['ProbU'],dframe['Tau'])# getting the tau of max count of ProbU
                # changing the couting to probabilities

                dframe['ProbU'] = np.vectorize(divide_arrays_by_cell, otypes=[object])(dframe['ProbU'], dframe['PindK'])
                dframe['PindU'] = np.vectorize(divide_arrays_by_cell_scalar, otypes=[object])(dframe['PindU'],
                                                                                              dframe['total'])
                dframe['ProbTl'] = np.vectorize(divide_arrays_by_cell, otypes=[object])(dframe['ProbTl'], dframe['total'])
                dframe['MI'] = np.vectorize(divide_arrays_by_cell, otypes=[object])(dframe['ProbU'], dframe['PindU'])
                dframe['HcondO'] = np.vectorize(remove_zero, otypes=[object])(dframe['HcondO'])
                # ************************************************************
                # applying the logarithm and the maximum between the result of
                # log for MI and 0 to avoid negative numbers
                dframe.MI = np.vectorize(logarithm_array, otypes=[object])(dframe['MI'])  # ,dframe['ProbTl'])
                # print(dframe.MI.values)
                # ************************************************************
                dframe['J_avg'] = dframe.J.apply(np.mean)
                dframe['O_sum'] = dframe.O.apply(np.sum)
                dframe['P_avg'] = dframe.P.apply(np.mean)
                dframe['Ou_sum'] = dframe.Ou.apply(np.sum)
                dframe['Pu_avg'] = dframe.Pu.apply(np.mean)

                # dframe['ProbTl_avg']=dframe.ProbTl.apply(np.mean)
                dframe['ProbU_avg'] = dframe.ProbU.apply(np.mean)
                dframe['HcondO_avg'] = dframe.HcondO.apply(np.mean)
                dframe['MI_avg'] = dframe.MI.apply(np.mean)  # <<<-----------
                dframe['Tau_avg'] = dframe.Tau.apply(np.mean)

                # dframe['ProbTl_max']=dframe.ProbTl.apply(np.max)
                dframe['ProbU_max'] = dframe.ProbU.apply(np.min)
                dframe['HcondO_max'] = dframe.HcondO.apply(np.min)
                dframe['MI_max'] = dframe.MI.apply(np.max)  # <<<-----------
                dframe['Tau_max'] = np.vectorize(tau_largest_count_to_surprisal_compute)(dframe['ProbU'], dframe[
                    'Tau'])  # getting the tau of max ProbU

                # dframe['HCONDO_group']=dframe['HCONDO']
                # dframe['MI_group']=dframe['HD']-dframe['HCONDO']

                dframe = dframe[dframe.user.isin(users_idx)]
                datagroup = dframe[cols1].describe(percentiles=[0.1, 0.5, 0.9]).values[1:, :]
                # print(dframe[cols1].describe(percentiles=[0.1,0.5,0.9]))
                # return

                users = np.array([str(fname)] * datagroup.shape[1]).reshape((1, datagroup.shape[1]))
                datagroup = np.append(users, datagroup, axis=0)
                # print(datagroup)
                # return datagroup
                for i in range(len(cols1)):
                    # print(i,datagroup[:,i])
                    ldatagroup[i].append(datagroup[:, i])

                idx = 0
                for fname2 in cols1:
                    dt = pd.DataFrame(np.array(ldatagroup[idx]), columns=col_st)
                    # print(fname2,'\n',dt[['user','avg','min','10%','50%','90%','max',]].round(2),'\n',dt.info())
                    # return

                    # if not 'user' in dict_dtype:
                    dict_dtype['user'] = str
                    dt = dt.astype(dict_dtype)  #
                    # print(fname2,'\n',dt[['user','avg','min','10%','50%','90%','max',]].round(2),'\n',dt.info())
                    # return

                    if not os.path.exists(STORE_PATH + '/stats-' + str(version) + '/' + fname2 + '-group.tsv'):
                        dt.to_csv(STORE_PATH + '/stats-' + str(version) + '/' + fname2 + '-group.tsv', sep='\t',
                                  header=True, float_format='%.3f', index=False)
                    else:
                        with open(STORE_PATH + '/stats-' + str(version) + '/' + fname2 + '-group.tsv', 'a') as f:
                            dt.to_csv(f, sep='\t', header=False, float_format='%.3f', index=False)
                    idx += 1

                ldatagroup = []
                for i in range(len(cols1)):
                    ldatagroup.append([])
            """
            -----------------------------------------------------------------
            """
            """
            checking and storing each user with minimum posting events
            """
            for user_id in tqdm(users_idx):
                # print('\n\t\tuser:',user_id)
                df = dframe[dframe.user == user_id].copy()
                # print(df.shape)

                if df.shape[0] >= 30:
                    for values in df.ProbU.values:
                        # print(values,type(values))#
                        if type(values) != np.ndarray:
                            values = [values]
                        # print(values,type(values),np.isnan(values))
                        pcvalues += list(values)

                    # for values in df.ProbTl.values:
                    #     if type(values) != np.ndarray:
                    #         values=[values]
                    #     prvalues+=list(values)

                    for tau in df.Tau.values:
                        if type(tau) != np.ndarray:
                            tau = [tau]
                        tauvalues += list(tau)

                    data2 = df[cols1].describe(percentiles=[0.1, 0.5, 0.9]).values[1:, :]
                    # print(df[cols1].describe(percentiles=[0.1,0.5,0.9]))
                    # return

                    users = np.array([str(user_id) + '-' + str(fname)] * data2.shape[1]).reshape((1, data2.shape[1]))
                    data2 = np.append(users, data2, axis=0)
                    # print(data2)
                    #                   return data2
                    for i in range(len(cols1)):
                        # print(i,data2[:,i])
                        ldata2[i].append(data2[:, i])
                    i = 0

                    data8 = df[cols1].corr('pearson')
                    data9 = df[cols1].corr('spearman')

                    ldata8[i].append(str(user_id) + '-' + str(fname))
                    ldata9[i].append(str(user_id) + '-' + str(fname))
                    i += 1
                    for row in range(1, len(cols1)):
                        for col in range(0, row):
                            ldata8[i].append(data8.values[row, col])
                            ldata9[i].append(data9.values[row, col])
                            i += 1
                    i = 0
                    """
                    Changing the metrics for probabilities
                    """
                    df.C = df.C / df.B
                    df.D = df.D / df.O_sum  #
                    df.F = df.F / TIME_WINDOW

                    # df.Tau_count=df.Tau_count/TIME_WINDOW
                    df.Tau_max = df.Tau_max / TIME_WINDOW
                    df.Tau_avg = df.Tau_avg / TIME_WINDOW

                    df.J_avg = df.J_avg / TIME_WINDOW
                    if df.D.isnull().any():
                        print()
                        print(df[df.D.isnull()])
                        print()
                        logger.log_warn(str(fname), str(df[df.D.isnull()].values))
                        df.D.fillna(0, inplace=True)

                    df.I = df.I / 5.  # df.B
                    df.H = df.H / 5.  # df.B

                    df.loc[df.B == 0, 'B'] = 1.
                    """
                    Converting probability zero to maximum surprisal (maximum entropy)
                    """
                    df.loc[df.C == 0, 'C'] = df.loc[df.C == 0, 'B']
                    df.loc[df.F == 0, 'F'] = TIME_WINDOW
                    df.loc[df.D == 0, 'D'] = df.loc[df.D == 0, 'O_sum']

                    df.loc[df.J_avg == 0, 'J_avg'] = TIME_WINDOW

                    df.loc[df.ProbU_avg == 0, 'ProbU_avg'] = 1.  ## the entropy of zero probability is zero by convention
                    # df.loc[df.ProbTl_avg == 0,'ProbTl_avg']=1. ## the entropy of zero probability is zero by convention
                    # df.loc[df.HcondO_avg == 0,'HcondO_avg']=1. ## the entropy of zero probability is zero by convention

                    # df.loc[df.MI_avg == 0,'MI_avg']=1. ## the entropy of zero probability is zero by convention
                    df.loc[df.Tau_avg == 0, 'Tau_avg'] = 1.

                    df.loc[df.ProbU_max == 0, 'ProbU_max'] = 1.  ## the entropy of zero probability is zero by convention
                    # df.loc[df.ProbTl_max == 0,'ProbTl_max']=1. ## the entropy of zero probability is zero by convention
                    # df.loc[df.HcondO_max == 0,'HcondO_max']=1. ## the entropy of zero probability is zero by convention
                    # df.loc[df.MI_max == 0,'MI_max']=1. ## the entropy of zero probability is zero by convention
                    df.loc[df.Tau_max == 0, 'Tau_max'] = 1.

                    # print('\n','-'*10,'\nShape:',df[df.Tau_avg == 0],'\n\n',df.Tau_avg.describe(),'\n','-'*10)

                    df.loc[df.H == 0, 'H'] = 1.
                    df.loc[df.I == 0, 'I'] = 1.
                    """
                    Changing the probabilities of metrics in bits via surprisal
                    """
                    df.B = np.log2(df.B)
                    df.C = np.abs(-np.log2(df.C))  # positive bits
                    df.D = np.abs(-np.log2(df.D))  # positive bits
                    df.F = np.abs(-np.log2(df.F))  # positive bits
                    df.J_avg = np.abs(-np.log2(df.J_avg))  # positive bits

                    df.H = -np.log2(df.H)
                    df.I = -np.log2(df.I)
                    df.P_avg = np.abs(-np.log2(df.P_avg))
                    df.Pu_avg = np.abs(-np.log2(df.Pu_avg))

                    df.ProbU_avg = np.abs(-np.log2(df.ProbU_avg))
                    # df.ProbTl_avg=np.abs(-np.log2(df.ProbTl_avg))
                    # df.HcondO_avg=np.abs(-np.log2(df.HcondO_avg))

                    # df.MI_avg=np.abs(np.log2(df.MI_avg)) # ====>>> NO NEGATIVE SIGNAL <<<==== !!!
                    df.Tau_avg = np.abs(-np.log2(df.Tau_avg))

                    df.ProbU_max = np.abs(-np.log2(df.ProbU_max))
                    # df.ProbTl_max=np.abs(-np.log2(df.ProbTl_max))
                    # df.HcondO_max=np.abs(-np.log2(df.HcondO_max))

                    # df.MI_max=np.abs(np.log2(df.MI_max)) # ====>>> NO NEGATIVE SIGNAL <<<==== !!!
                    df.Tau_max = np.abs(-np.log2(df.Tau_max))
                    # --------------------------------------------------------------
                    df.loc[df.B == 0, 'B'] = 0.
                    df.loc[df.C == 0, 'C'] = 0.
                    df.loc[df.D == 0, 'D'] = 0.
                    df.loc[df.F == 0, 'D'] = 0.
                    df.loc[df.H == 0, 'H'] = 0.
                    df.loc[df.I == 0, 'I'] = 0.
                    df.loc[df.J_avg == 0, 'J_avg'] = 0.

                    df.loc[df.ProbU_avg == 0, 'ProbU_avg'] = 0.
                    # df.loc[df.ProbTl_avg == 0, 'ProbTl_avg']=0.
                    # df.loc[df.HcondO_avg == 0, 'HcondO_avg']=0.

                    # df.loc[df.MI_avg == 0, 'MI_avg']=0.
                    df.loc[df.Tau_avg == 0, 'Tau_avg'] = 0.

                    df.loc[df.ProbU_max == 0, 'ProbU_max'] = 0.
                    # df.loc[df.ProbTl_max == 0, 'ProbTl_max']=0.
                    # df.loc[df.HcondO_max == 0, 'HcondO_max']=0.

                    # df.loc[df.MI_max == 0, 'MI_max']=0.
                    df.loc[df.Tau_max == 0, 'Tau_max'] = 0.
                    """
                    --------------------------------------------------------------- 
                    """
                    # users_idx=df.groupby(by=['user']).count().index.values
                    # print(users_idx)
                    if df[df.user.isin([user_id])].shape[0] >= 30:
                        cols_store = ['user', 'C', 'D', 'F', 'G', 'Gu', 'H', 'I', 'J_avg', 'P_avg', 'Pu_avg', \
                                      'ProbU_avg', 'MI_super_avg', 'MI_avg', 'Tau_avg', \
                                      'ProbU_max', 'MI_super_max', 'MI_max', 'Tau_max', \
                                      'HcondO_avg', 'HcondO_max', 'L', 'timestamp']  # 'HCONDO_group','MI_group',\
                        # 'ProbTl_avg','ProbTl_max',
                        dt = df[df.user.isin([user_id])][cols_store].copy()
                        dt['group'] = str(user_id) + '-' + str(fname)
                        cols_store = ['group'] + cols_store
                        dt[cols_store].to_csv(STORE_PATH + '/users/' + str(user_id) + '-' + str(fname) + '.tsv', sep='\t', \
                                              index=False, header=True, float_format='%.3f')

                    data1 = df[cols1].describe(percentiles=[0.1, 0.5, 0.9]).values[1:, :]

                    users = np.array([str(user_id) + '-' + str(fname)] * data1.shape[1]).reshape((1, data1.shape[1]))
                    data1 = np.append(users, data1, axis=0)

                    for i in range(len(cols1)):
                        ldata[i].append(data1[:, i])
                    i = 0

                    data10 = df[cols1].corr('pearson')
                    data11 = df[cols1].corr('spearman')

                    ldata10[i].append(str(user_id) + '-' + str(fname))
                    ldata11[i].append(str(user_id) + '-' + str(fname))
                    i += 1
                    for row in range(1, len(cols1)):
                        for col in range(0, row):
                            ldata10[i].append(data10.values[row, col])
                            ldata11[i].append(data11.values[row, col])
                            i += 1

                    logger.log_done(str(user_id) + '-' + str(fname))
                    records += 1

                    if records % RECORDS_PERIOD == 0:
                        # print("RECORDS: ",records)
                        idx = 0
                        for fname2 in cols1:
                            dt = pd.DataFrame(np.array(ldata[idx]), columns=col_st)
                            dt = dt.astype(dict_dtype)  # {'user': str})

                            if not os.path.exists(STORE_PATH + '/stats-' + str(version) + '/' + fname2 + '.tsv'):
                                dt.to_csv(STORE_PATH + '/stats-' + str(version) + '/' + fname2 + '.tsv', sep='\t',
                                          header=True, float_format='%.3f', index=False)
                            else:
                                with open(STORE_PATH + '/stats-' + str(version) + '/' + fname2 + '.tsv', 'a') as f:
                                    dt.to_csv(f, sep='\t', header=False, float_format='%.3f', index=False)
                            idx += 1
                        idx = 0
                        for fname2 in cols1:
                            dt = pd.DataFrame(np.array(ldata2[idx]), columns=col_st)
                            dt = dt.astype(dict_dtype)  # {'user': str})

                            if not os.path.exists(STORE_PATH + '/stats-' + str(version) + '/' + fname2 + '-raw.tsv'):
                                dt.to_csv(STORE_PATH + '/stats-' + str(version) + '/' + fname2 + '-raw.tsv', sep='\t',
                                          header=True, float_format='%.3f', index=False)
                            else:
                                with open(STORE_PATH + '/stats-' + str(version) + '/' + fname2 + '-raw.tsv', 'a') as f:
                                    dt.to_csv(f, sep='\t', header=False, index=False, float_format='%.3f')
                            idx += 1
                        records = 0

                        ldata = []
                        for i in range(len(cols1)):
                            ldata.append([])
                        ldata2 = []
                        for i in range(len(cols1)):
                            ldata2.append([])

                        list_dt = [ldata10, ldata11, ldata8, ldata9]
                        list_cols = [cols_corrs, cols_corrs]

                        recording_files(list_dt, list_cols, STORE_PATH)

                        ldata10, ldata11 = [], []
                        for i in range(len(cols_corrs)):
                            ldata10.append([])
                            ldata11.append([])

                        ldata8, ldata9 = [], []
                        for i in range(len(cols_corrs)):
                            ldata8.append([])
                            ldata9.append([])
                else:
                    logger.log_warn(str(user_id) + '-' + str(fname), 'user with posting not enought...')
                    # logger.log_warn(str(fname), 'group with users not enought...')
        #                break
        # except Exception as e:
        #     print(e,row,col)
        #     print()
        #     logger.log_warn(str(fname), e)
        # return
        else:
            print('\n\t', fname, 'not processed by curious model linear\n')


    dp = pd.DataFrame(pcvalues)
    # dpr=pd.DataFrame(prvalues)
    dtau = pd.DataFrame(tauvalues)

    # if not os.path.exists(STORE_PATH+'/stats-'+str(version)+'/data_probr.tsv'):
    #     dpr.to_csv(STORE_PATH+'/stats-'+str(version)+'/data_probr.tsv',index=False,header=False,sep='\t',float_format='%.2f')
    # else:
    #     with open(STORE_PATH+'/stats-'+str(version)+'/data_probr.tsv','a') as f:
    #         dpr.to_csv(f,sep='\t',header=False,float_format='%.2f',index=False)

    if not os.path.exists(STORE_PATH + '/stats-' + str(version) + '/data_prob_cond.tsv'):
        dp.to_csv(STORE_PATH + '/stats-' + str(version) + '/data_prob_cond.tsv', index=False, header=False, sep='\t',
                  float_format='%.2f')
    else:
        with open(STORE_PATH + '/stats-' + str(version) + '/data_prob_cond.tsv', 'w') as f:
            dp.to_csv(f, sep='\t', header=False, float_format='%.2f', index=False)

    if not os.path.exists(STORE_PATH + '/stats-' + str(version) + '/data_tau.tsv'):
        dtau.to_csv(STORE_PATH + '/stats-' + str(version) + '/data_tau.tsv', index=False, header=False, sep='\t',
                    float_format='%.2f')
    else:
        with open(STORE_PATH + '/stats-' + str(version) + '/data_tau.tsv', 'w') as f:
            dtau.to_csv(f, sep='\t', header=False, float_format='%.2f', index=False)

    idx = 0
    if ldata[0] != []:
        for fname in cols1:
            dt = pd.DataFrame(np.array(ldata[idx]), columns=col_st)
            dt = dt.astype(dict_dtype)  # {'user': str})
            if not os.path.exists(STORE_PATH + '/stats-' + str(version) + '/' + fname + '.tsv'):
                dt.to_csv(STORE_PATH + '/stats-' + str(version) + '/' + fname + '.tsv', sep='\t', header=False,
                          float_format='%.2f', index=False)
            else:
                with open(STORE_PATH + '/stats-' + str(version) + '/' + fname + '.tsv', 'a') as f:
                    dt.to_csv(f, sep='\t', header=False, float_format='%.2f', index=False)
            idx += 1

    idx = 0
    if ldata2[0] != []:
        for fname in cols1:
            dt = pd.DataFrame(np.array(ldata2[idx]), columns=col_st)
            dt = dt.astype(dict_dtype)  # {'user': str})
            if not os.path.exists(STORE_PATH + '/stats-' + str(version) + '/' + fname + '-raw.tsv'):
                dt.to_csv(STORE_PATH + '/stats-' + str(version) + '/' + fname + '-raw.tsv', sep='\t', header=False,
                          float_format='%.2f', index=False)
            else:
                with open(STORE_PATH + '/stats-' + str(version) + '/' + fname + '-raw.tsv', 'a') as f:
                    dt.to_csv(f, sep='\t', header=False, float_format='%.2f', index=False)
            idx += 1

    if ldata10[0] != [] and ldata11[0] != [] and ldata8[0] != [] and ldata9[0] != []:
        list_dt = [ldata10, ldata11, ldata8, ldata9]
        list_cols = [cols_corrs, cols_corrs]
        recording_files(list_dt, list_cols, STORE_PATH)


def multiprocessing_bits(logger, path, STORE_PATH, index_slices, version):
    block = []
    for slices in index_slices:
        blk = Process(target=working_process_bits, \
                      args=(logger, slices, path, STORE_PATH, version,))
        blk.start()
        print('%d,%s' % (blk.pid, " starting..."))
        block.append(blk)

    for blk in block:
        print('%d,%s' % (blk.pid, " waiting..."))
        blk.join()


if __name__ == '__main__':
    if not os.path.exists(os.getcwd() + "/logs/"):
        os.makedirs(os.getcwd() + "/logs/")
    logger = Logger(os.getcwd() + "/logs")

    STORE_PATH = 'stimulus-bits/'
    # path='/data/users/amsousa/Whatsapp/data/'
    path = './'

    if not os.path.exists(path + STORE_PATH):
        os.makedirs(path + STORE_PATH)

    dgroup_info = pd.read_csv(path + 'dataset_gname_raw_info.csv', sep='\t', encoding='utf-8')
    # dgroup_info=pd.read_csv(path+'dataset_gname_raw_info.csv',sep='\t', encoding='utf-8')

    N_PROC = 60
    groups_idx = dgroup_info.gname.values  # groupID.values#gname.values
    # groups_idx=dgroup_info.gname.values
    SIZE = groups_idx.size
    print('SIZE: ', SIZE)
    lenght = int(np.ceil(SIZE / N_PROC))
    shares = [i * lenght for i in range(N_PROC)]
    index_slices = []
    for i in range(N_PROC):
        if i < N_PROC - 1:
            index_slices.append(groups_idx[shares[i]:shares[i + 1]])
        else:
            index_slices.append(groups_idx[shares[i]:])

            # index_slices=[['556592507094-152086396']]
    # index_slices=[['13134264254-154144618']]
    # index_slices=[['17379733320-154096056']]
    # index_slices=[['554599947927-151959352']]
    # index_slices=[['5511949722517-152830261']]
    # index_slices=[['558586062371-149072446']]
    # index_slices=[['558585910888-148770249']]
    version = 1
    print('USERS: %d' % SIZE)
    multiprocessing_bits(logger, path, path + STORE_PATH, index_slices, version)