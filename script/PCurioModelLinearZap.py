#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 13:26:02 2018

@author: alexandre
"""

import os
import copy
import glob
import datetime
import json as js
from datetime import timezone
import numpy as np
import time as Time
import pandas as pd
# import pandas as pd
from logger import Logger
from util import binary_search
from multiprocessing import Process, Queue, current_process
from characterizaton import stats_probabilities
from tqdm import tqdm

from numba import njit
from numba.core import types
from numba.typed import Dict, List


def check_percentiles(list_percentiles, dict_percentiles, percentile, time, \
                      total_users, dmp, fname):
    for j in range(len(list_percentiles)):
        if total_users == list_percentiles[j] and total_users != 0:
            if not percentile[j] in dict_percentiles:
                # -------------------------------------------------------------
                dmp_copy = copy.deepcopy(dmp)
                # dmp_new=dmp_copy
                # dmp_new={str(k): {str(k2):v2 for k2,v2 in dmp_copy[k].items()} 
                #               if type(v)!=int
                #               else v  for k,v in dmp_copy.items()}                 
                # print(js.dumps(dmp_new,indent=2))
                # -------------------------------------------------------------
                dict_col = {}
                for row in dmp_copy:
                    if row != 'total':
                        for col in dmp_copy[row].keys():
                            if col != 'degree_output' and col != 'total_col' and \
                                col != 'degree_input' and col != 'total_row' and \
                                col != 'time':
                                if col in dict_col:
                                    dict_col[col] += dmp_copy[row][col]['num']
                                else:
                                    dict_col[col] = dmp_copy[row][col]['num']
                for col in dict_col:
                    if not col in dmp_copy:
                        dmp_copy[col] = {}
                        dmp_copy[col]['time'] = time
                        dmp_copy[col]['total_row'] = 0
                        dmp_copy[col]['total_col'] = 0
                        dmp_copy[col]['degree_input'] = 0
                        dmp_copy[col]['degree_output'] = 0
                    else:
                        dmp_copy['total'] -= dmp_copy[col]['total_col']
                    dmp_copy[col]['total_col'] = dict_col[col]
                dmp_copy['total'] = np.sum(list(dict_col.values()))
                # print(dmp_copy['total'])
                # -------------------------------------------------------------                    
                dst_row, dst_col, lpr_cond_o, lpr_cond_d, lpr_joint, lpr_marg_o, \
                    lpr_marg_d, size_users = stats_probabilities(dmp_copy, fname)
                # print('Marg_o:')
                # print(lpr_marg_o)
                # print('lpr_cond_o:')
                # print(lpr_cond_o)
                # print('lpr_cond_d:')
                # print(lpr_cond_d)
                # print('lpr_joint:')
                # print(lpr_joint)
                if np.any(lpr_marg_o < 0):
                    # print(js.dumps(dmp_new,indent=2))
                    return
                HO = np.abs(-np.sum(lpr_marg_o * np.log2(lpr_marg_o)))
                HCOND_O = np.abs(-np.sum(lpr_joint * np.log2(lpr_cond_o)))
                HCOND_D = np.abs(-np.sum(lpr_joint * np.log2(lpr_cond_d)))
                MI = HO - HCOND_D  # HJOINT-HCOND_O-HCOND_D
                # -------------------------------------------------------------
                dict_percentiles[percentile[j]] = {}
                dict_percentiles[percentile[j]]['time'] = time
                dict_percentiles[percentile[j]]['MI'] = MI
                dict_percentiles[percentile[j]]['HCOND_O'] = HCOND_O
    return dict_percentiles


@njit
def compute_expectation_entropy_numba(mtx_prob, lrows):
    dict_hcond_o = Dict.empty(
        key_type=types.int64,
        value_type=types.float64,
    )
    for row in lrows:
        lcols = np.where(mtx_prob[row] != 0)[0]
        if lcols.size > 0:
            for col in lcols:
                if row != col and mtx_prob[row][col] != 0:
                    pr = mtx_prob[row][col] / (mtx_prob[row].sum() - mtx_prob[row][row])
                    value = -1 * pr * np.log2(pr)
                    if row in dict_hcond_o:
                        dict_hcond_o[row] += value
                    else:
                        dict_hcond_o[row] = value
    return dict_hcond_o


def compute_expectation_entropy(dmp, lrows):
    dict_hcond_o = {}

    for row in lrows:
        if row in dmp.keys() and row in dmp[row].keys():
            dmp[row]['total_row'] -= dmp[row][row]['num']

    for row in lrows:  # dmp.keys():
        if row in dmp.keys():
            for col in dmp[row].keys():
                if col != 'degree_output' and col != 'total_col' and \
                    col != 'degree_input' and col != 'total_row' and \
                    col != 'time':

                    if row != col:
                        pr = dmp[row][col]['num'] / dmp[row]['total_row']

                        # ----- Group conditional expectation on O = o --------
                        value = pr * np.log2(pr)  # <<<<----------------
                        if row in dict_hcond_o:
                            dict_hcond_o[row] += value
                        else:
                            dict_hcond_o[row] = value
                        # -----------------------------------------------------   

    for key in dict_hcond_o.keys():
        dict_hcond_o[key] = -1 * dict_hcond_o[key]

    return dict_hcond_o


@njit
def compute_partial_entropy_numba(Us, mtx_prob):
    HcondO = List()
    PO = List()
    Hd_partial = List()
    for row in Us:
        # row=dkey[u]
        lcols = np.where(mtx_prob[row] != 0)[0]
        if lcols.size > 0:
            total_row = mtx_prob[row].sum() - mtx_prob[row][row]
            if total_row <= 0:
                continue
            total = mtx_prob.sum() - np.diag(mtx_prob).sum()

            Hd = 0.
            for col in lcols:
                if col != row:
                    total_col = mtx_prob[col].sum() - mtx_prob[col][col]
                    if total_col <= 0:
                        continue
                    elem = mtx_prob[row][col]
                    pj = elem / total
                    pd = total_col / total
                    Hd += pj * np.log2(pd)

            Hd = -Hd
            if Hd == 0:
                Hd = 0
            Hd_partial.append(Hd)

            pr = mtx_prob[row][lcols[lcols != row]] / total_row
            po = total_row / total
            h = -1 * np.sum(pr * np.log2(pr))
            if h == 0:
                h = 0

            HcondO.append(h)
            PO.append(po)

    return HcondO, Hd_partial, PO


@njit
def update_contingency_table_numba(queue_user_all, queue_time_all, vector_count, vector_queue_time, \
                                   mtx_prob, mtx_time, Dt):
    n_tam = len(queue_user_all)
    for index in range(n_tam):
        u = queue_user_all[index]
        time = queue_time_all[index]

        vector_count[u] += 1
        if vector_queue_time[u][0] == 0:
            vector_queue_time[u][0] = time
        else:
            vector_queue_time[u].append(time)

        lindex = np.where(mtx_time != 0)[0]
        if lindex.size > 0:
            for k in lindex:
                if time - mtx_time[k] <= Dt:
                    while (time - vector_queue_time[k][0] > Dt):
                        vector_queue_time[k].pop(0)
                        vector_count[k] -= 1
                    if k != u:
                        n = vector_count[k]
                        mtx_prob[k][u] += n
                else:
                    vector_count[k] = 1
                    vector_queue_time[k] = List([time])

        mtx_time[u] = time

    return mtx_prob, mtx_time


def add_dict(d, key):
    if key in d.keys():
        d[key] += 1
    else:
        d[key] = 1


def add_dict_time(d, d2, key, value):
    if key in d.keys():
        d2[key] = d[key]
    else:
        d2[key] = value
    d[key] = value


def get_user_genres(message, urls, imageID, videoID, audioID):
    user_media = []
    if message != ' ' and message != '' \
        and message != 'None' and message != 'one':
        user_media.append(0)
    if len(urls) >= 1:
        if len(urls[0]) > 0:
            user_media.append(1)
    if imageID != '' and imageID != 'empty' and len(imageID) > 0:
        user_media.append(2)
    if videoID != '' and videoID != 'empty' and len(videoID) > 0:
        user_media.append(3)
    if audioID != '' and audioID != 'empty' and len(audioID) > 0:
        user_media.append(4)

    url_number = 0
    if len(urls) >= 1:
        for u in urls:
            if len(u) > 0:
                # print('URL:',u,'lenght: ',len(u))
                url_number += 1

    return user_media, url_number


@njit
def get_list_from_dict(l, d):
    lvalues = List()
    for k in l:
        lvalues.append(d[k])
    return lvalues


def matrix_to_dict_json(mtx_prob, dkey):
    dict_mtx_prob = {"total": int(mtx_prob.sum() - np.diag(mtx_prob).sum())}

    for row in range(mtx_prob.shape[0]):
        user_number_row = dkey[row]
        dict_mtx_prob[str(user_number_row)] = {}
        # ---------------------------------------------------------------------
        discount = mtx_prob[row][row]
        dict_mtx_prob[str(user_number_row)]['total_row'] = \
            mtx_prob[row].sum() - discount
        dict_mtx_prob[str(user_number_row)]['total_col'] = \
            mtx_prob[:, row].sum() - discount
        # ---------------------------------------------------------------------
        dict_mtx_prob[str(user_number_row)]['degree_input'] = \
            mtx_prob[:, row][mtx_prob[:, row] != 0].size
        dict_mtx_prob[str(user_number_row)]['degree_output'] = \
            mtx_prob[row][mtx_prob[row] != 0].size
        # ---------------------------------------------------------------------
        for col in range(mtx_prob.shape[1]):
            if row != col and mtx_prob[row][col] != 0:
                user_number_col = dkey[col]

                dict_mtx_prob[str(user_number_row)][str(user_number_col)] = {}
                dict_mtx_prob[str(user_number_row)][str(user_number_col)]['num'] = \
                    int(mtx_prob[row][col])
        # ---------------------------------------------------------------------        
    return dict_mtx_prob


def computeStimulusDegree(fin, fout, group_id, window, path_mtx_prob, fname):
    if not os.path.exists("./logs/"):
        os.makedirs("./logs/")
    logger = Logger("./logs/")
    logger.add_log({'empty': 'key.empty'})
    fout.write(
        "#user\tA\tB\tC\tD\tE\tF\tG\tGu\tH\tI\tL\tM\tN\tJ\tO\tP\tMu\tOu\tPu\tNu\tProbTl\tPindK\tPindU\tProbU\tTauU\tPO\tHd_partial\tHcondO\tHCONDO\tHD\tTotalU\tUs\tTotal\ttimestamp\n")
    # fout.write("#user\tA\tB\tC\tD\tE\tF\tG\tGu\tH\tI\tL\tM\tN\tJ\tO\tP\tMu\tOu\tPu\tNu\tProbTl\tPindK\tPindU\tProbU\tTauU\tHcondO\tUs\tTotal\ttimestamp\n")
    count = 0
    TIME_WINDOW = window * 1800  # 3600
    LOWER_BOUND_EVENTS = 10
    last_time = 0

    queue_user = List()  ##[]#queue_art=[]
    queue_media = []  # queue_genres=[]
    queue_time = List()  ##[]
    dict_user = {}  # dict_art={}
    dict_media = {}  # dict_genres={}
    duser = {}  # dart={}
    dmedia = {}  ##dgenres={}
    lduser = []  # ldart=[]
    ldmedia = []  # ldgenres=[]

    dict_user_time = {}
    dict_media_time = {}
    duser_time = {}
    dmedia_time = {}
    lduser_time = []
    ldmedia_time = []
    """
    ---------------------------------------------------------------------------
    """
    queue_user_all = List()
    queue_time_all = List()

    Dt = 30 * 60
    # dict_mtx_prob={'total':0}
    # dict_count={}
    """
    --------------------------------------------------------------------------
    """
    dkey = Dict.empty(
        key_type=types.int64,
        value_type=types.int64,
    )
    dkey_convert = Dict.empty(
        key_type=types.int64,
        value_type=types.int64,
    )
    for i, k in enumerate(fin.usernumber.unique()):
        dkey[k] = i
        dkey_convert[i] = k

    n_size = fin.usernumber.unique().size
    mtx_prob = np.zeros([n_size, n_size])
    mtx_time = np.zeros(n_size)
    vector_count = List()  # np.zeros(n_size)
    vector_queue_time = List([])
    for i in range(n_size):
        vector_queue_time.append(List([0.]))
        vector_count.append(0)
    """
    ---------------------------------------------------------------------------
    """
    # print(fin.shape) 
    store_line = ''
    for idx in tqdm(fin.index):
        # try:
        # print(count)#,end=',')
        # if count==60:
        #    break
        count += 1
        _ = fin['username'][idx]
        _ = fin['groupId'][idx]
        _ = fin['groupname'][idx]
        timestamp = fin['timestamp'][idx]
        user = fin['usernumber'][idx]
        """
        *********** Media's Categorie ******************
        """
        user_media = [fin['message_category'][idx]]

        # if len(user_media) == 0:
        #     continue
        #     raise Exception("Post without media's category...")
        """
        ************************************************
        """
        time_date = datetime.datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S%z')
        time_date = time_date.replace(tzinfo=timezone.utc)
        """
        -------------------------------------------------------------------
        """
        # if count == 1:
        #    time_begin = time_date
        """
        -------------------------------------------------------------------
        """
        size = len(queue_time)

        if size > 0:
            last_time = datetime.datetime.utcfromtimestamp(queue_time[-1])
            last_time = last_time.replace(tzinfo=timezone.utc)
        else:
            last_time = time_date

        total_sec = time_date - last_time

        if size == 0 or total_sec.total_seconds() <= TIME_WINDOW:
            # time_seconds=Time.mktime(time_date.timetuple())
            time_seconds = datetime.datetime.timestamp(time_date)
            queue_user.append(user)
            queue_time.append(time_seconds)  #
            queue_media.append(user_media)

            add_dict(dict_user, user)

            for media in user_media:
                add_dict(dict_media, media)

            lduser.append(dict_user.copy())
            ldmedia.append(dict_media.copy())
            """
            ***************************************************************
            """
            add_dict_time(dict_user_time, duser_time, user, time_seconds)
            for media in user_media:
                add_dict_time(dict_media_time, dmedia_time, media, time_seconds)

            lduser_time.append(dict_user_time.copy())
            ldmedia_time.append(dict_media_time.copy())
            """
            ***************************************************************
            """
            if size > 2:
                i = binary_search(queue_time, TIME_WINDOW)
                if i > 0:
                    """
                    -------------------------------------------------------
                    """
                    queue_user_all = queue_user[:i].copy()  # concatenate
                    queue_time_all = queue_time[:i].copy()  # concatenate
                    # -----------------------------------------------------
                    queue_users_all_cpy = get_list_from_dict(queue_user_all, dkey)
                    mtx_prob, mtx_time = \
                        update_contingency_table_numba(queue_users_all_cpy, \
                                                       queue_time_all, vector_count, vector_queue_time, \
                                                       mtx_prob, mtx_time, Dt)

                    del queue_time_all
                    del queue_user_all
                    """
                    -------------------------------------------------------
                    """
                    del queue_user[:i]
                    del queue_media[:i]
                    del queue_time[:i]

                    for key in lduser[i - 1].keys():
                        if key in dict_user.keys():
                            duser[key] = lduser[i - 1][key]

                    for key in ldmedia[i - 1].keys():
                        if key in dict_media.keys():
                            dmedia[key] = ldmedia[i - 1][key]
                    # ------------------------------------------------------
                    for key in lduser_time[i - 1].keys():
                        if key in dict_user_time.keys():
                            duser_time[key] = lduser_time[i - 1][key]

                    for key in ldmedia_time[i - 1].keys():
                        if key in dict_media_time.keys():
                            dmedia_time[key] = ldmedia_time[i - 1][key]
                    # ------------------------------------------------------
                    del lduser[:i]
                    del ldmedia[:i]
                    # ------------------------------------------------------
                    del lduser_time[:i]
                    del ldmedia_time[:i]
        else:
            #                time_seconds=Time.mktime(time_date.timetuple())
            time_seconds = datetime.datetime.timestamp(time_date)
            """
            ---------------------------------------------------------------
            """
            queue_user_all = queue_user.copy()  # concatenate
            queue_time_all = queue_time.copy()  # concatenate
            # -------------------------------------------------------------
            queue_users_all_cpy = get_list_from_dict(queue_user_all, dkey)
            mtx_prob, mtx_time = \
                update_contingency_table_numba(queue_users_all_cpy, \
                                               queue_time_all, vector_count, vector_queue_time, \
                                               mtx_prob, mtx_time, Dt)

            del queue_time_all
            del queue_user_all
            """
            --------------------------------------------------------------
            """
            queue_time = List([time_seconds])
            queue_user = List([user])
            queue_media = [user_media]
            dict_user = {}
            dict_media = {}
            duser = {}
            dmedia = {}
            lduser = []
            ldmedia = []
            # --------------------------------------------------------------
            dict_user_time = {}
            dict_media_time = {}
            duser_time = {}
            dmedia_time = {}
            lduser_time = []
            ldmedia_time = []
            # --------------------------------------------------------------
            add_dict(dict_user, user)
            for media in user_media:
                add_dict(dict_media, media)

            lduser.append(dict_user.copy())
            ldmedia.append(dict_media.copy())
            # --------------------------------------------------------------
            add_dict_time(dict_user_time, duser_time, user, time_seconds)
            for media in user_media:
                add_dict_time(dict_media_time, dmedia_time, media, time_seconds)

            lduser_time.append(dict_user_time.copy())
            ldmedia_time.append(dict_media_time.copy())
        # --------------------------------------------------------------

        size = len(queue_time)
        if size >= LOWER_BOUND_EVENTS:  # storing the records in the file!!!
            # print('Registering...')
            # time_seconds=Time.mktime(time_date.timetuple())
            time_seconds = datetime.datetime.timestamp(time_date)
            A = count
            B = size

            if user in dict_user.keys():
                if not user in duser.keys():
                    C = dict_user[user] - 1  # len(set(queue_art))
                else:
                    C = dict_user[user] - duser[user] - 1
            else:
                C = 0

            if user in dict_user_time.keys():
                if user in duser_time.keys():
                    fraction = (dict_user_time[user] - duser_time[user])
                    if fraction < TIME_WINDOW:
                        if fraction == 0:
                            fraction = TIME_WINDOW
                        F = int(TIME_WINDOW - fraction)
                    else:
                        F = 0
                else:
                    F = 0
            else:
                F = 0

            J = []
            for k in user_media:  # dict_media_time.keys():
                if k in dmedia_time.keys():
                    fraction = (dict_media_time[k] - dmedia_time[k])
                    if fraction < TIME_WINDOW:
                        if fraction == 0:
                            fraction = TIME_WINDOW
                        J.append(int(TIME_WINDOW - fraction))
                    else:
                        J.append(0)
                else:
                    J.append(0)
            J = np.array(J, dtype=int)

            # arr_user=np.array(queue_user)
            # E=url_number#compute_cache_hit_entropy_2(arr_user,user)

            E = 0

            freq_media = []
            for media in user_media:
                if media in dict_media.keys():
                    freq_media.append(media)
            D = [dict_media[k] - dmedia[k]
                 if k in dict_media.keys() and k in dmedia.keys()
                 else dict_media[k]
                 for k in freq_media]
            D = np.mean(D)
            # D=url_number

            I = len(user_media)  # instantaneous complexity

            L = user_media  # str(topic)#topic of current user's message
            N = []  # list of medias from history
            for k in dict_media.keys():
                if not k in dmedia.keys():
                    if dict_media[k] != 0:
                        N.append(str(k))
                elif dict_media[k] - dmedia[k] != 0:
                    N.append(str(k))

            O = [dict_media[k] - dmedia[k]
                 if k in dmedia.keys()
                 else dict_media[k]
                 for k in dict_media.keys()]

            Ou = [dict_user[k] - duser[k]
                  if k in duser.keys()
                  else dict_user[k]
                  for k in dict_user.keys()]

            Us = []
            ProbU = []
            TauU = []
            ProbTl = []  # joint probability of user and k had posted in delta t time unit
            PindU = []  # probability of user
            PindK = []  # probability of any k user had posted with
            HcondO = []

            for k in dict_user.keys():
                if k in duser.keys():
                    if k != user and not k in Us and dict_user[k] - duser[k] >= 0:
                        row = dkey[k]
                        col = dkey[user]
                        if mtx_prob[row][col] != 0:
                            Us.append(k)
                            # conditional probability: Pr(D=d|O=o)
                            p = mtx_prob[row][col]
                            ProbU.append(p)
                            # joint probability: Pr(D=d,O=o)
                            ptl = mtx_prob[row][col]
                            ProbTl.append(ptl)
                            # marginals probabilities
                            discount = mtx_prob[row][row]

                            pk = mtx_prob[row].sum() - discount
                            PindK.append(pk)

                            discount = mtx_prob[col][col]
                            pu = mtx_prob[:, col].sum() - discount
                            PindU.append(pu)

                            TauU.append(0.)
                        else:
                            ProbU.append(0)
                            ProbTl.append(0)
                            PindK.append(0)
                            PindU.append(0)
                            TauU.append(0.)
                    else:
                        ProbU.append(0)
                        ProbTl.append(0)
                        PindK.append(0)
                        PindU.append(0)
                        TauU.append(0.)
            # -------------------------------------------------------------
            Hd_partial = []
            PO = []
            # verifica se existem valores válidos na matriz de contingência
            SIZE = np.where(mtx_prob.sum(axis=0) != 0)[0]

            if SIZE.size > 0 and len(Us) > 0:
                Us_typed = List()
                [Us_typed.append(u) for u in Us]
                Us_typed_2 = get_list_from_dict(Us_typed, dkey)
                HcondO, Hd_partial, PO = compute_partial_entropy_numba(Us_typed_2, mtx_prob)

            if len(Us) > 0 and len(HcondO) == 0:
                for u in Us:
                    Hd_partial.append(0)
                    HcondO.append(0)
            # -------------------------------------------------------------
            total_mtx_prob = mtx_prob.sum() - np.diag(mtx_prob).sum()
            ProbU = np.array(ProbU)

            if len(Us) == 0:
                ProbU = [0]
                ProbTl = [0]
                PindU = [0]
                PindK = [0]
                TauU = [0]
                HcondO = [0]
                PO = [0]
                Hd_partial = [0]
                Us = [0]

            """
            *********** To compute group's entropy each step **************
            """
            MI, HCONDO = 0, 0  # compute_entropies(time_seconds,dict_mtx_prob,fname)
            HD = MI + HCONDO
            """
            ***************************************************************
            """

            Pu = np.array(Ou, dtype=int)
            Pu = Pu[Pu != 0]
            Ou = Pu
            Hu = Pu.size
            summation = Pu.sum()
            Pu = Pu / summation
            h = -Pu * np.log2(Pu)
            Gu = h.sum()
            # print("Probabilities: ",Pu.round(2),Gu)

            P = np.array(O, dtype=int)
            P = P[P != 0]
            O = P
            H = P.size  # overall complexity
            M = H  # number of topics in window
            summation = P.sum()
            P = P / summation
            h = -P * np.log2(P)
            G = h.sum()
            store_line += '%s\t%d\t%d\t%d\t%.2f\t%d\t%d\t%.2f\t%.2f\t%d\t%d\t' % \
                          (user, A, B, C, D, E, F, G, Gu, H, I)  # A: number of line; B: total number of itens;
            store_line += ','.join(('%d' % (l)) for l in L) + '\t'  # medias of current user
            store_line += '%d\t' % (M)  # this component do inform the size if next vector
            store_line += ','.join(N) + '\t'  # medias of access's historical
            store_line += ','.join(('%d' % (j)) for j in J) + '\t'  # latest timestamp of medias of historical
            store_line += ','.join(('%d' % (o)) for o in O) + '\t'  # counting of medias from historical
            store_line += ','.join(('%.2f' % (p)) for p in P) + '\t'  # +'\n'# probability of each media of historical
            store_line += '%d\t' % (Hu)  # this component do inform the size if next vector
            store_line += ','.join(('%d' % (o)) for o in Ou) + '\t'  # counting of users from historical
            store_line += ','.join(('%.2f' % (p)) for p in Pu) + '\t'  # +'\n'# probability of each user of historical
            store_line += '%d\t' % (len(Us))  # size of vector of users from conditional probability of posting
            store_line += ','.join(('%d' % (pt)) for pt in ProbTl) + '\t'  # JOINT probability of every user
            store_line += ','.join(('%d' % (pk)) for pk in PindK) + '\t'  # INDIVIDUAL probability of k had posted
            store_line += ','.join(('%d' % (pu)) for pu in PindU) + '\t'  # INDIVIDUAL probability of user
            store_line += ','.join(
                ('%d' % (pc)) for pc in ProbU) + '\t'  # conditional probability of user to post after any other user u
            store_line += ','.join(
                ('%.2f' % (tau)) for tau in TauU) + '\t'  # interposting time of conditional probability
            store_line += ','.join(('%.2f' % (po)) for po in PO) + '\t'  # Pr(O=o) weighted
            store_line += ','.join(('%.2f' % (hd)) for hd in Hd_partial) + '\t'  #
            store_line += ','.join(
                ('%.2f' % (hcond_o)) for hcond_o in HcondO) + '\t'  # expectation entropy conditioned on origin
            store_line += '%.2f\t' % (HCONDO)  # conditional entropy (general)
            store_line += '%.2f\t' % (HD)  # entropy of destination
            store_line += '%d\t' % (np.where(mtx_prob.sum(axis=0))[0].size)  # quantity of current users
            store_line += ','.join(('%d' % (us)) for us in Us) + '\t'  # user's id of conditional probability
            store_line += '%d\t' % (total_mtx_prob)
            store_line += str(time_seconds) + '\n'
            # -------------------------------------------------------------------------
            # fout.write(store_line)
            # fout.flush()
            # store_line=''
            # -------------------------------------------------------------------------

        # if count == 10000:
        #     break

    # except Exception as e:
    #     print(str(e)+" line: "+str(count))
    #     logger.log_warn(str(group_id), str(e)+" line: "+str(count))

    dmp = matrix_to_dict_json(mtx_prob, dkey_convert)
    # ---> print(js.dumps(dmp,indent=4))
    with open(path_mtx_prob + str(group_id) + '.txt', 'w') as json_file:
        js.dump(dmp, json_file, indent=4)

    fout.write(store_line)
    fout.flush()
    logger.log_done(str(group_id))


def parsing_date(timestamp):
    ts = timestamp[:19]
    return ts


"""
def working_process(path,slices,window):    
    count=0
    path_validation="/stimulus-filter-validation-model-1/"
    for fname in slices:
        print('\t%s'%(fname))

        lpath_time=glob.glob(path + 'groups-validation-model-1/'+fname.split('/')[0]+'/*')
        
        for path_time in lpath_time:
            f=fname.split('/')[0]
            fin=pd.read_csv(path_time+'/'+f+'.tsv',keep_default_na=False, sep='\t',\
                           encoding='utf-8')
            fin.urls=fin.urls.str.replace("[",'')
            fin.urls=fin.urls.str.replace("]",'')
            fin.urls=fin.urls.str.replace("\'",'')
            fin.urls=fin.urls.str.split(',')
            #fin['timestamp']=pd.to_datetime(fin['timestamp'])
            fin.sort_values(by=['timestamp'],ascending=True,inplace=True)  
            
            fin.timestamp=fin.timestamp.apply(parsing_date)
            
            f=path_time.split('/')
            path_timestamp=f[-1]
            fname=f[-2]
    
            if not os.path.exists(path+path_validation+'/'+fname+'/'+path_timestamp+'/'):
                os.makedirs(path+path_validation+'/'+fname+'/'+path_timestamp+'/')
            fout=open(path+path_validation+'/'+fname+'/'+path_timestamp+'/'+fname+'.txt','a')
            
            path_mtx_prob=path+path_validation+"/group_mtx_prob/"+fname+'/'+path_timestamp+'/'
            if not os.path.exists(path_mtx_prob):
                os.makedirs(path_mtx_prob)
    
            computeStimulusDegree(fin,fout,fname,window,path_mtx_prob,fname)
        
            fout.close()
            count+=1
"""


def working_process(path, slices, window):
    count = 0
    for fname in slices:
        fin = pd.read_csv(path + 'groups/' + str(fname) + '.tsv', keep_default_na=False, sep='\t', encoding='utf-8')

        print('\t%s (%d lines)' % (fname, len(fin.index)))
        fin.sort_values(by=['timestamp'], ascending=True, inplace=True)

        if not os.path.exists(path + "/stimulus-filter/"):
            os.makedirs(path + "/stimulus-filter/")

        if not os.path.exists(path + "/stimulus-filter/" + str(fname) + '.txt'):
            fout = open(path + "/stimulus-filter/" + str(fname) + '.txt', 'w')

            # matriz de probabilidade
            path_mtx_prob = path + "/stimulus-filter/group_mtx_prob/"
            if not os.path.exists(path_mtx_prob):
                os.makedirs(path_mtx_prob)

            computeStimulusDegree(fin, fout, fname, window, path_mtx_prob, fname)

            fout.close()
        else:
            print("Already done")

        count += 1


def multiprocessing(path, index_slices, window):
    block = []
    for slices in index_slices:
        blk = Process(target=working_process, args=(path, slices, window,))
        blk.start()
        print('%d,%s' % (blk.pid, " starting..."))
        block.append(blk)

    for blk in block:
        print('%d,%s' % (blk.pid, " waiting..."))
        blk.join()


def init_proccess(path, window, queue):
    while not queue.empty():
        try:
            working_process(path, [queue.get()], window)
        except queue.Empty:
            break

    print("Finishing process %d" % current_process().pid)


def multiprocess_manager(path, groups, window, N_PROC):
    queue = Queue()
    block = []

    for group in groups:
        queue.put(group)

    for proc in range(N_PROC):
        blk = Process(target=init_proccess, args=(path, window, queue))
        blk.start()
        print('Process %d: %d,%s' % (proc, blk.pid, " starting..."))
        block.append(blk)

    for blk in block:
        print('%d,%s' % (blk.pid, " waiting..."))
        blk.join()


if __name__ == '__main__':
    path = './'

    dgroup_info = pd.read_csv(path + 'dataset_gname_raw_info.csv', sep='\t', encoding='utf-8')

    groups_idx = dgroup_info.gname.values

    lf = glob.glob('./data/stimulus-filter/*.txt')
    groups_idx_done = []

    for f in lf:
        groups_idx_done.append(f[len('./data/stimulus-filter/'):-4])

    groups_idx_done = np.array(groups_idx_done)
    if len(groups_idx_done) != 0:
        print(groups_idx_done[0], groups_idx_done.size)
    else:
        print('\nNothing was done!....\n')
    groups_idx = np.array(list(set(groups_idx).difference(set(groups_idx_done))))

    N_PROC = 34

    print("Starting...")

    multiprocess_manager(path, groups_idx, 1, N_PROC)

    """

    SIZE = groups_idx.size
    lenght = int(np.ceil(SIZE / N_PROC))
    shares = [i * lenght for i in range(N_PROC)]
    index_slices = []
    for i in range(N_PROC):
        if i < N_PROC - 1:
            index_slices.append(groups_idx[shares[i]:shares[i + 1]])
        else:
            index_slices.append(groups_idx[shares[i]:])

    print('Starting...')
    for window in [1]:  # [4,8,12,16,20,24]:
        start = Time.time()
        # Para execução de um grupo
        # working_process(path,['553384008305-153426695'],window)
        multiprocessing(path, index_slices, window)
        end = Time.time()
        runtime = end - start
        print("\nRuntime: ", runtime)

    """
