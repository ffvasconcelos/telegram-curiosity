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
from multiprocessing import Process
from characterizaton import stats_probabilities


def check_percentiles(list_percentiles, dict_percentiles, percentile, time, total_users, dmp, fname):
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


def update_contingency_table(queue_user_all, queue_time_all, dict_count, dict_mtx_prob, Dt, list_percentiles, dict_percentiles, fname):
  dict_mtx_prob_col = {}
  dict_mtx_prob_output_degree = {}
  # -------------------------------------------------------------------------
  """
  percentile=[10,50,90]
  total_users=0
  for key in dict_mtx_prob:
      if key != 'total' and dict_mtx_prob[key]['total_row'] == 0 and\
          dict_mtx_prob[key]['total_col'] == 0:
              total_users+=1
  total_users=len(dict_mtx_prob)-total_users-1# subtract the counting of 'total' key
  # print('---> total_users: ',total_users)
  time=0
  for key in dict_mtx_prob:
      if key != 'total':
          if time < dict_mtx_prob[key]['time']:
              time = dict_mtx_prob[key]['time']
  # print(total_users,time,queue_time_all[0])
  if total_users > 0:
      # print('------> total_users: ',total_users)
      dict_percentiles=check_percentiles(list_percentiles,\
                              dict_percentiles,percentile,time,total_users,\
                                                         dict_mtx_prob,fname)
  """
  # -------------------------------------------------------------------------
  for index in range(len(queue_user_all)):
    u = queue_user_all[index]
    time = queue_time_all[index]
    """
    print('-'*50)
    print('Index: ',index+1,'\tu:',u,'\t time:',time,'\n')
    print('\ndict_mtx_prob:\n',js.dumps(dict_mtx_prob,indent=4),'\n')
    """
    if u in dict_count.keys():
      dict_count[u]['count'] += 1
      dict_count[u]['queue'].append(time)
    else:
      dict_count[u] = {}
      dict_count[u]['count'] = 1
      dict_count[u]['queue'] = [time]
    # print('\ndict_count:\n',js.dumps(dict_count,indent=4))
    for key in dict_mtx_prob.keys():
      """
      print('key: ',key,'\t t - t_last: ',\
            time - dict_mtx_prob[key]['time'],'\t Dt: ',Dt)
      """
      if (key != 'total') and (time - dict_mtx_prob[key]['time'] <= Dt):
        # tau=time-dict_mtx_prob[key]['time']
        # print('Tau: ',tau)
        while (time - dict_count[key]['queue'][0] > Dt):
          dict_count[key]['queue'].pop(0)
          dict_count[key]['count'] -= 1
        # print('\ndict_count:\n',js.dumps(dict_count,indent=4))
        tau_array = np.array(dict_count[key]['queue'])
        tau_array = time - tau_array
        tau_array = tau_array[tau_array != 0]

        if not u in dict_mtx_prob[key]:
          # ---------------------------------------------------------
          """
          if u != key: # they can not be the same users!
              total_users+=1 # add the new user 'u' (destiny)       
              # print('total_users: ',total_users)
              if np.sum(np.array(list_percentiles) == 0) != 3:
                  dict_percentiles=check_percentiles(list_percentiles,\
                      dict_percentiles,percentile,time,total_users,\
                           dict_mtx_prob,fname)
                  # print('entering 1')
              if dict_mtx_prob[key]['degree_output'] == 0 and\
                  dict_mtx_prob[key]['total_row'] == 0:
                      total_users+=1 # add the user 'key' (origin)
                      if np.sum(np.array(list_percentiles) == 0) != 3:
                          dict_percentiles=check_percentiles(list_percentiles,\
                            dict_percentiles,percentile,time,total_users,\
                               dict_mtx_prob,fname)
                      # print('entering....')
              # print('entering....')
          """
          # ---------------------------------------------------------
          dict_mtx_prob[key][u] = {}
          dict_mtx_prob[key][u]['num'] = 0
          dict_mtx_prob[key][u]['tau'] = 0.
          dict_mtx_prob[key]['degree_output'] += 1
          if u in dict_mtx_prob_output_degree:
            dict_mtx_prob_output_degree[u] += 1  # update the output degree
          else:
            dict_mtx_prob_output_degree[u] = 1

        if key == u:
          n = dict_count[key]['count'] - 1
        else:
          n = dict_count[key]['count']

        dict_mtx_prob[key][u]['num'] += n
        n_arr = dict_mtx_prob[key][u]['num']
        dict_mtx_prob[key][u]['tau'] *= n_arr - n
        dict_mtx_prob[key][u]['tau'] += np.float(tau_array.sum())
        dict_mtx_prob[key][u]['tau'] /= n_arr

        dict_mtx_prob[key]['total_row'] += n
        if u in dict_mtx_prob_col.keys():
          dict_mtx_prob_col[u] += n
        else:
          dict_mtx_prob_col[u] = n
          # update the input degree of user key
        """
        print('\ntime - queue_time:  ',time,dict_count[key]['queue'],\
             tau_array,n_arr)  
        """
      elif (key != 'total'):
        # reseting dictionary of counting and queue
        dict_count[key]['count'] = 1
        dict_count[key]['queue'] = [time]

    if u in dict_mtx_prob.keys():
      dict_mtx_prob[u]['time'] = time
    else:
      dict_mtx_prob[u] = {}
      dict_mtx_prob[u]['time'] = time
      dict_mtx_prob[u]['total_row'] = 0
      dict_mtx_prob[u]['total_col'] = 0
      dict_mtx_prob[u]['degree_input'] = 0
      dict_mtx_prob[u]['degree_output'] = 0
  for key in dict_mtx_prob_col.keys():
    dict_mtx_prob[key]['total_col'] += dict_mtx_prob_col[key]
    if key in dict_mtx_prob_output_degree:
      dict_mtx_prob[key]['degree_input'] += dict_mtx_prob_output_degree[key]
    dict_mtx_prob['total'] += dict_mtx_prob_col[key]
  return dict_mtx_prob, dict_count


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


def computeStimulusDegree(fin, fout, group_id, window, path_mtx_prob, fname):
  if not os.path.exists(os.getcwd() + "/logs/"):
    os.makedirs(os.getcwd() + "/logs/")
  logger = Logger(os.getcwd() + "/logs/")
  logger.add_log({'empty': 'key.empty'})
  fout.write(
    "#user\tA\tB\tC\tD\tE\tF\tG\tGu\tH\tI\tL\tM\tN\tJ\tO\tP\tMu\tOu\tPu\tNu\tProbTl\tPindK\tPindU\tProbU\tTauU\tHcondO\tUs\tTotal\ttimestamp\n")
  count = 0
  TIME_WINDOW = window * 1800  # 3600
  LOWER_BOUND_EVENTS = 10
  last_time = 0

  queue_user = []  # queue_art=[]
  queue_media = []  # queue_genres=[]
  queue_time = []
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
  queue_user_all = []
  queue_time_all = []

  Dt = 30 * 60
  dict_mtx_prob = {'total': 0}
  dict_count = {}
  dict_percentiles = {}  # dictionary to record percentiles of user number
  """
  ---------------------------------------------------------------------------
  """
  # print(fin.shape)
  store_line = ''
  for idx in fin.index:
    try:
      # print(count,end=',')
      # if count==60:
      #    break
      count += 1
      _ = fin['username'][idx]
      _ = fin['groupID'][idx]
      _ = fin['groupname'][idx]
      timestamp = fin['timestamp'][idx]
      user = fin['usernumber'][idx]
      """
      *********** Media's Categorie ******************
      """
      user_media = fin['message_category'][idx]

      # if len(user_media) == 0:
      #  continue
      #  raise Exception("Post without media's category...")
      """
      ************************************************
      """
      time_date = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
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
      # print(count,user,timestamp,user_media,total_sec.total_seconds())
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
            """
            print('Line: ',count,'total: ',dict_mtx_prob['total'])
            #print('matrix:\n',dict_mtx_prob,'\n')   
            dmp={str(k):{str(k2):v2 for k2,v2 in dmp[k].items()} 
                 if type(v)!=int else v  for k,v in dmp.items()}                    
            #print(dmp.keys(),'\n')
            print(json.dumps(dmp, indent=4))
            print('\n','-'*100,'\n')
            """
            # -----------------------------------------------------
            # print('\nthen...update_contingency_table\n')
            dict_mtx_prob, dict_count = \
              update_contingency_table(queue_user_all, queue_time_all, \
                                       dict_count, dict_mtx_prob, Dt, list_percentiles, \
                                       dict_percentiles, fname)
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
        """
        print('Line: ',count,'total: ',dict_mtx_prob['total'])
        #print('matrix:\n',dict_mtx_prob,'\n')
        dmp={str(k): {str(k2):v2 for k2,v2 in dmp[k].items()} 
             if type(v)!=int else v  for k,v in dmp.items()}                    
        #print(dmp.keys(),'\n')
        print(json.dumps(dmp, indent=4))
        print('\n','-'*100,'\n')
        """
        # -------------------------------------------------------------
        # print('\nelse...update_contingency_table\n')
        dict_mtx_prob, dict_count = \
          update_contingency_table(queue_user_all, queue_time_all, \
                                   dict_count, dict_mtx_prob, Dt, list_percentiles, \
                                   dict_percentiles, fname)
        del queue_time_all
        del queue_user_all
        """
        --------------------------------------------------------------
        """
        queue_time = [time_seconds]
        queue_user = [user]
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
      """
      print('SIZE OF QUEUE:%d'%size)
      print(queue_time)
      print(dict_user)
      print(duser)
      print(lduser,'\n')
      print('User:\n',dict_user_time)
      print(duser_time,'\n')
      print(lduser_time)
      print(queue_user)
      print('Media:\n',dict_media_time)
      print(dmedia_time)
      print(ldmedia_time)
      print(queue_media)
      print()
      """
      if size >= LOWER_BOUND_EVENTS:  # storing the records in the file!!!
        # print('Registering...')
        #                time_seconds=Time.mktime(time_date.timetuple())
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
              F = np.int(TIME_WINDOW - fraction)
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
              J.append(np.int(TIME_WINDOW - fraction))
            else:
              J.append(0)
          else:
            J.append(0)
        J = np.array(J, dtype=np.int)

        # arr_user=np.array(queue_user)
        # E = url_number  # compute_cache_hit_entropy_2(arr_user,user)
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

        # print('Line: ',A,'\n')
        for k in dict_user.keys():
          if k in duser.keys():
            if k != user and not k in Us and dict_user[k] - duser[k] >= 0:
              Us.append(k)
              if k in dict_mtx_prob.keys():
                if user in dict_mtx_prob[k].keys():
                  # conditional probability: Pr(D=d|O=o)
                  p = dict_mtx_prob[k][user]['num']  # /dict_mtx_prob[k]['total_row']
                  ProbU.append(p)
                  # joint probability: Pr(D=d,O=o) NOTE:
                  ptl = dict_mtx_prob[k][user]['num']
                  ProbTl.append(ptl)
                  # marginals probabilities
                  # ----------------------------------------
                  discount = 0
                  if k in dict_mtx_prob[k]:
                    discount = dict_mtx_prob[k][k]['num']
                  # ----------------------------------------
                  pk = dict_mtx_prob[k]['total_row'] - discount  # /dict_mtx_prob['total']
                  PindK.append(pk)
                  # ----------------------------------------
                  discount = 0
                  if user in dict_mtx_prob[user]:
                    discount = dict_mtx_prob[user][user]['num']
                  # ----------------------------------------
                  pu = dict_mtx_prob[user]['total_col'] - discount  # /dict_mtx_prob['total']
                  PindU.append(pu)
                  # interposting time average
                  tau = dict_mtx_prob[k][user]['tau']
                  TauU.append(tau)
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
        if len(Us) != 0:
          # compute the expectation entropy conditioned on origin
          # (influencers from window)
          dmp_copy = copy.deepcopy(dict_mtx_prob)
          dict_hcond_o = compute_expectation_entropy(dmp_copy, Us)
          for k in Us:
            if k in dict_hcond_o.keys():
              HcondO.append(dict_hcond_o[k])
            else:
              HcondO.append(0)
        # -------------------------------------------------------------
        # total summation of contingency matrix
        sum_discount = 0
        for row in dict_mtx_prob:
          if row != 'total' and row in dict_mtx_prob[row]:
            sum_discount += dict_mtx_prob[row][row]['num']
        total_mtx_prob = dict_mtx_prob['total'] - sum_discount
        ProbU = np.array(ProbU)

        if len(Us) != len(ProbU) or len(Us) != len(TauU) or len(Us) != len(ProbTl):
          raise Exception('conditional probability: different size between user and probabitlity array...')

        if len(Us) == 0:
          ProbU = [0]
          ProbTl = [0]
          PindU = [0]
          PindK = [0]
          TauU = [0]
          HcondO = [0]
          Us = [0]

        Pu = np.array(Ou, dtype=np.int)
        Pu = Pu[Pu != 0]
        Ou = Pu
        Hu = Pu.size
        summation = Pu.sum()
        Pu = Pu / summation
        h = -Pu * np.log2(Pu)
        Gu = h.sum()
        # print("Probabilities: ",Pu.round(2),Gu)

        P = np.array(O, dtype=np.int)
        P = P[P != 0]
        O = P
        H = P.size  # overall complexity
        M = H  # number of topics in window
        summation = P.sum()
        P = P / summation
        h = -P * np.log2(P)
        G = h.sum()
        store_line += '%s\t%d\t%d\t%d\t%.2f\t%d\t%d\t%.2f\t%.2f\t%d\t%d\t' % \
                      (user, A, B, C, D, E, F, G, Gu, H, I)
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
        store_line += ','.join(('%.2f' % (tau)) for tau in TauU) + '\t'  # interposting time of conditional probability
        store_line += ','.join(
          ('%.2f' % (hcond_o)) for hcond_o in HcondO) + '\t'  # expectation entropy conditioned on origin
        store_line += ','.join(('%d' % (us)) for us in Us) + '\t'  # user's id of conditional probability
        store_line += '%d\t' % (total_mtx_prob)
        store_line += str(time_seconds) + '\n'
        # dmp=dict_mtx_prob.copy()
        # dmp={str(k): {str(k2):v2 for k2,v2 in dmp[k].items()}
        #     if type(v)!=int else v  for k,v in dmp.items()}
        # print(dmp.keys(),'\n')
        # print(json.dumps(dmp, indent=4))
        # print('\n','-'*100,'\n')
        # -------------------------------------------------------------------------
        # fout.write(store_line)
        # fout.flush()
        # store_line=''
        # -------------------------------------------------------------------------
    except Exception as e:
        print(str(e)+" line: "+str(count))
        logger.log_warn(str(group_id), str(e)+" line: "+str(count))

  dmp = dict_mtx_prob.copy()
  dmp = {str(k): {str(k2): v2 for k2, v2 in dmp[k].items()}
  if type(v) != int
  else v for k, v in dmp.items()}
  # writing the json file
  with open(path_mtx_prob + str(group_id) + '.txt', 'w') as json_file:
    js.dump(dmp, json_file, indent=4)
  # -------------------------------------------------------------------------
  """
  if np.sum(np.array(list_percentiles) == 0) != 3:
      dpercentiles=pd.DataFrame(dict_percentiles)# recording the dict percentiles
      if not os.path.exists(path_mtx_prob[:-15]+'/percentiles/'):
          os.makedirs(path_mtx_prob[:-15]+'/percentiles/')
      # print(list_percentiles)
      # for key in dpercentiles:
      #     print(key,'\t',dpercentiles[key].values[0])
      # writing the percentiles files
      if not 50 in dpercentiles.keys() or not 90 in dpercentiles.keys():
          print(fname,'--------',dpercentiles.keys())
      dpercentiles.to_csv(path_mtx_prob[:-15]+'/percentiles/'+group_id+'.tsv',\
                          sep='\t',header=True,index=True)
  """
  # -------------------------------------------------------------------------
  """    
  print(js.dumps(dmp,indent=4))
  
  print('Total Summation: ',dmp['total'])
  s=0
  for key in dmp.keys():
      if key != 'total':
def divide_arrays_by_cell(arr1,arr2):
  i=np.nonzero(arr2)[0]
  #print(arr1,arr1.dtype,arr2,arr2.dtype)
  if i.size == 0:
      result=np.zeros(arr1.shape,dtype=np.float)
  else:
      result=arr1[i]/arr2[i]
  #print(result,result.dtype)
  result=result[np.nonzero(result)]
  if result.size == 0:
      result=np.array([0])
  return result
          total_row=dmp[key]['total_row']
          s+=total_row
          if total_row != 0:
              total_row
              #print('%s:\t%d\t%2.8f'%(key,total_row,total_row/dmp['total']))
  print('TOTAL summation: ',s)
  print('number of users:',len(list(dmp.keys())))
  
  #print(js.dumps(dmp['557194028836'],indent=4))
  """
  logger.log_done(str(group_id))
  fout.write(store_line)
  fout.flush()


def working_process(path, slices, window):
  count = 0
  for fname in slices:
    print('\t%s' % (fname))
    fin = pd.read_csv(path + 'groups/' + str(fname) + '.tsv', keep_default_na=False, sep='\t', encoding='utf-8')
    fin.sort_values(by=['timestamp'], ascending=True, inplace=True)

    if not os.path.exists(path + "/stimulus-filter/"):
      os.makedirs(path + "/stimulus-filter/")
    fout = open(path + "/stimulus-filter/" + str(fname) + '.txt', 'a')

    # matriz de probabilidade
    path_mtx_prob = path + "/stimulus-filter/group_mtx_prob/"
    if not os.path.exists(path_mtx_prob):
      os.makedirs(path_mtx_prob)

    computeStimulusDegree(fin, fout, fname, window, path_mtx_prob, fname)

    fout.close()
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

  N_PROC = 2
  SIZE = groups_idx.size
  lenght = np.int(np.ceil(SIZE / N_PROC))
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
