#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:59:10 2020

@author: alexandre
"""
import os
import glob
import json as js
import numpy as np
import pandas as pd
import networkx as nx

from collections import Counter
from logger import Logger
from scipy import stats as st
from scipy.stats import ks_2samp

from matplotlib import pyplot as plt
from biokit.viz import corrplot

from util import plot_cdf_list_curves
# from ContingencyMatrix import compute_graph_measures

def stats_probabilities(dmp,fname):
    dict_st_row={}
    dict_st_col={}
    lpr_cond_o=[]
    lpr_cond_d=[]
    lpr_joint=[]
    lpr_marg_o=[]
    lpr_marg_d=[]
    marg_d={}
    marg_o={}
    
    for row in dmp.keys():
        if row != 'total':
            for col in dmp[row].keys():
                if col != 'degree_output' and col != 'total_col' and\
                   col != 'degree_input'  and col != 'total_row' and\
                   col != 'time':
                    if row == col:
                        dmp[row]['total_row']-=dmp[row][col]['num']
                        dmp[row]['total_col']-=dmp[row][col]['num']
                        dmp['total']-=dmp[row][col]['num']
                        dmp[row]['degree_input']-=1
                        dmp[row]['degree_output']-=1            
    
    for row in dmp.keys():
        if row != 'total':
            prob_max=-1.
            prob_min=10.0
            tau_max=-1
            tau_min=-1
            for col in dmp[row].keys():
                if col != 'degree_output' and col != 'total_col' and\
                   col != 'degree_input'  and col != 'total_row' and\
                   col != 'time':
                    # if row == col:   
                    #     dmp[row]['total_row']-=dmp[row][col]['num']
                    #     dmp[row]['total_col']-=dmp[row][col]['num']
                    #     dmp['total']-=dmp[row][col]['num']
                    #     dmp[row]['degree_input']-=1
                    #     dmp[row]['degree_output']-=1
                    if row != col:
                        pr=dmp[row][col]['num']/dmp[row]['total_row']
                        lpr_cond_o.append(pr)
                        tau=dmp[row][col]['tau']
                        if pr >= prob_max:
                            prob_max=pr
                            tau_max=tau
                        if pr <= prob_min:
                            prob_min=pr
                            tau_min=tau
                        
                        pj=dmp[row][col]['num']
                        lpr_joint.append(pj)
                        if not col in marg_d:
                            marg_d[col]=dmp[row][col]['num']
                        else:
                            marg_d[col]+=dmp[row][col]['num']
                        
                        pc=dmp[row][col]['num']/dmp[col]['total_col']
                        lpr_cond_d.append(pc)
                        if col in dict_st_col:
                            if pc >= dict_st_col[col]['max']:
                                dict_st_col[col]['max']=pc
                                dict_st_col[col]['max_tau']=tau
                            if pc <= dict_st_col[col]['min']:
                                dict_st_col[col]['min']=pc
                                dict_st_col[col]['min_tau']=tau
                        else:
                            dict_st_col[col]={}
                            dict_st_col[col]['max']=pc
                            dict_st_col[col]['min']=pc
                            dict_st_col[col]['max_tau']=tau
                            dict_st_col[col]['min_tau']=tau
                            
            if dmp[row] != {}:
                if prob_min != 10 or prob_max != -1:
                    dict_st_row[row]={}
                if prob_max != -1:
                    dict_st_row[row]['max']=prob_max
                    dict_st_row[row]['max_tau']=tau_max
                if prob_min != 10:
                    dict_st_row[row]['min']=prob_min
                    dict_st_row[row]['min_tau']=tau_min
                if dmp[row]['total_row'] != 0:
                    po=dmp[row]['total_row']
                    # print(po)
                    lpr_marg_o.append(po)
                    if not row in marg_o:
                        marg_o[row]=po
    
    lpr_joint=np.array(lpr_joint)
    lpr_joint=lpr_joint/dmp['total']
    
    lpr_cond_o=np.array(lpr_cond_o)
    lpr_cond_d=np.array(lpr_cond_d)
    
    lpr_marg_o=np.array(lpr_marg_o)/dmp['total']
    
    lpr_marg_d=np.array(list(marg_d.values()))
    lpr_marg_d=lpr_marg_d[lpr_marg_d!=0]/dmp['total']
    
    if lpr_joint[lpr_joint == 0].size > 0:
        raise Exception('error prob zero....')
    
    dst_row=pd.DataFrame(dict_st_row)
    dst_row=dst_row.T
    dst_row['group']=fname
    dst_col=pd.DataFrame(dict_st_col)
    dst_col=dst_col.T
    dst_col['group']=fname
    
    size_users=len(set(marg_d.keys()).union(set(marg_o)))
    
    return dst_row,dst_col,\
        lpr_cond_o,lpr_cond_d,lpr_joint,lpr_marg_o,lpr_marg_d,\
            size_users

if __name__ == "__main__":
    
    if not os.path.exists(os.getcwd() + "/logs/"):
        os.makedirs(os.getcwd() + "/logs/")
    logger = Logger(os.getcwd() + "/logs")    
    
    path='./data/stimulus-filter-1-10-social-entropy-30-v4/group_mtx_prob/'    
    fnames=glob.glob(path+'*.txt')
    
    dg_filtering=pd.read_csv('./data/group_filtering.tsv',sep='\t')
    
    lfnames=[]
    for gid in dg_filtering['group-id'].values:
        for f in fnames:
            if gid in f:
                lfnames.append(f)
        
    dst_col=pd.DataFrame()
    dst_row=pd.DataFrame()
    
    lhcond_o=[]
    lhcond_d=[]
    lmi=[]
    lh_o=[]
    lh_d=[]
    
    lass=[]
    lclu=[]
    lrec=[]
    lscc=[]
    lden=[]
    
    lgroupID=[]
    lsize_users=[]

    # fnames=[path+'5511996290419-153625485.txt']
    # fnames=[]
    # fnames+=[path+'/917574994707-147586154.txt'] # 
    # fnames=[path+'/553194547170-153832231.txt'] #
    # fnames=[path+'/5527998351493-151096491.txt'] # 
    # fnames+=[path+'/556186017552-152764171.txt'] # 
    # fnames=[path+'/557182140907-151411832.txt'] 
    # fnames+=[path+'/19017083924-153870785.txt']     
    # fnames=[path+'/554391977545-149039957.txt']   
    # fnames+=[path+'/5521973612191-152104629.txt']   
    # fnames+=[path+'/557798089302-151595165.txt']   
    # fnames=[path+'/554599947927-151959352.txt']
    # fnames=[path+'/5522974001949-152486841.txt']
    # fnames=[path+'/556399951697-152744777.txt']
    # fnames=[path+'/557491383688-152726679.txt']
    # lfnames=[path+'/923450373132-152484416.txt']
    # lfnames=[path+'/553194547170-153832231.txt']
    # lfnames=[path+'/558388269209-152517858.txt']
    
    # Examples of characterization:
    # lfnames=[path+'/5521981539368-153722974.txt'] # CLUSTER 0
    # lfnames=[path+'/557491430779-150758866.txt'] # CLUSTER 0
    # lfnames=[path+'/19017083924-153870785.txt'] # CLUSTER 1
    # lfnames=[path+'/556496072212-147920838.txt'] # CLUSTER 1
    # lfnames=[path+'/13134264254-154144618.txt'] # CLUSTER 3
    # lfnames=[path+'/554198467628-151847233.txt'] # CLUSTER 2
    # lfnames=[path+'/18137219380-147846301.txt'] # CLUSTER 4
    # lfnames=[path+'/553891713089-153332026.txt'] # CLUSTER 5
    
    # lfnames=[path+'/5511996644954-153226711.txt']
    label_cluster=3
    
    count=0
    for f in lfnames:
        count+=1
        print(count,f[64:-4],end=': ')
        
        with open(f) as json_file:
            dict_mtx_prob = js.load(json_file)
            #print(js.dumps(dict_mtx_prob,indent=4))
        dmp=dict_mtx_prob.copy()
        

        dsr,dsc,lpr_cond_o,lpr_cond_d,lpr_joint,lpr_marg_o,lpr_marg_d,size_users=\
                                              stats_probabilities(dmp,f[64:-4])
        # break
        # in_degrees,out_degrees,pr_top_one_id,reciprocity,assortativity,\
        #     clustering,density,scc_size=compute_graph_measures(dmp)
        
        # lass.append(assortativity)
        # lclu.append(clustering)
        # lrec.append(reciprocity)
        # lscc.append(scc_size)
        # lden.append(density)
        
        # print('\n\tRec: %.2f\tAss: %.2f\tClus: %.2f\tDen: %.2f\tSCC: %.2f'%\
        #   (reciprocity,assortativity,clustering,density,scc_size))        
        
        dst_row=dst_row.append(dsr)
        dst_col=dst_col.append(dsc)
        
        HO=np.abs(-np.sum(lpr_marg_o*np.log2(lpr_marg_o)))
        HD=np.abs(-np.sum(lpr_marg_d*np.log2(lpr_marg_d)))
        HCOND_O=np.abs(-np.sum(lpr_joint*np.log2(lpr_cond_o)))
        HCOND_D=np.abs(-np.sum(lpr_joint*np.log2(lpr_cond_d)))
        HJOINT=np.abs(-np.sum(lpr_joint*np.log2(lpr_joint)))
        MI=HO-HCOND_D#HJOINT-HCOND_O-HCOND_D
        
        if HCOND_O <= HD and HCOND_D <= HO:
            print('\tHCOND_O: %.2f\tHCOND_D %.2f\tHD: %.2f\tHO: %.2f\tMI: %.2f\tUSERS: %d'%\
                  (HCOND_O,HCOND_D,HD,HO,MI,size_users))
        else:
            print('\tERROR: %.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%d'%\
                  (HCOND_O,HCOND_D,HD,HO,MI,HJOINT,size_users))
        
        lhcond_d.append(HCOND_D)
        lhcond_o.append(HCOND_O)
        lmi.append(MI)
        lh_o.append(HO)
        lh_d.append(HD)
        
        lgroupID.append(f[64:-4])
        lsize_users.append(size_users)
        # break
    
    plot_cdf_list_curves([lpr_cond_o],\
                          [r'$Pr(D=d|O=o)$'],'',\
                          SET_TITLE=True,LEG_LOC='best',SET_LOG=False,\
                          FONT_SIZE_LEG=12,FIG_SIZE_X=6,FIG_SIZE_Y=5,CCDF=False,\
                          SET_LEG=False,title_name='Cluster %d: %s'%\
                          (label_cluster,f[65:-4]),XLIM=False,\
                              SET_GRID=False,SET_MARKER=True)    
    
    dentropy=pd.DataFrame(data=np.array([lhcond_o,lhcond_d,lh_d,lh_o,lmi]).T,\
                                         # lass,lclu,lrec,lscc,lden]).T,
                          columns=['HCOND_O','HCOND_D','HD','HO','MI'])#,\
    # 'assort','cluster','recip','scc','dens'])
    corr=dentropy[['HCOND_O','HCOND_D','MI']].corr('pearson').round(2)
    c = corrplot.Corrplot(corr)
    c.plot()
    
    dentropy['groupID']=lgroupID
    dentropy['users']=lsize_users
    # dentropy.to_csv('./data/cluster-entropy/entropy.tsv',sep='\t',\
    #                 header=True,index=False)
        
    lrow=[]
    for key in ['max','min']:#,'self']:
        # values=-np.log2(dst_row[key].values.copy())
        values=dst_row[key].values.copy()
        lrow.append(values)
           
    lcol=[]
    for key in ['max','min']:
        # values=-np.log2(dst_col[key].values.copy())
        values=dst_col[key].values.copy()
        lcol.append(values)
        
    plot_cdf_list_curves(lcol+lrow,\
                         [r'$max\{Pr(O=o|D=d)$',r'$min\{Pr(O=o|D=d)\}$',\
                          r'$max\{Pr(D=d|O=o)$',r'$min\{Pr(D=d|O=o)\}$'],\
                         'Conditional probabilities',\
                         LEG_LOC='best',SET_LOG=False,FONT_SIZE_LEG=12,\
                         FIG_SIZE_X=6,FIG_SIZE_Y=5,CCDF=False,\
                         XLIM=False,SET_GRID=False,SET_MARKER=True)
    
    lrow_tau=[]
    for key in ['max_tau','min_tau']:#,'self']:
        # values=-np.log2(dst_row[key].values.copy())
        values=dst_row[key].values.copy()
        lrow_tau.append(values/60)
           
    lcol_tau=[]
    for key in ['max_tau','min_tau']:
        # values=-np.log2(dst_col[key].values.copy())
        values=dst_col[key].values.copy()
        lcol_tau.append(values/60)
        
    plot_cdf_list_curves(lcol_tau+lrow_tau,\
                         [r'$max\{Pr(O=o|D=d)$',r'$min\{Pr(O=o|D=d)\}$',\
                          r'$max\{Pr(D=d|O=o)$',r'$min\{Pr(D=d|O=o)\}$'],\
                         'Time in minutes',\
                         LEG_LOC='best',SET_LOG=False,FONT_SIZE_LEG=12,\
                         FIG_SIZE_X=6,FIG_SIZE_Y=5,\
                         XLIM=False,SET_GRID=False,SET_MARKER=True)        
    
    plot_cdf_list_curves([lhcond_o.copy(),lhcond_d.copy(),lmi.copy(),\
                          lh_o.copy(),lh_d.copy()],\
                         [r'$H(D|O)$',r'$H(O|D)$',\
                          r'$MI(O;D)$',r'$H(O)$',r'$H(D)$'],\
                         'Uncertainty in bits',\
                         LEG_LOC='best',SET_LOG=False,FONT_SIZE_LEG=12,\
                         FIG_SIZE_X=6,FIG_SIZE_Y=5,\
                         XLIM=False,SET_GRID=False,SET_MARKER=True)      

    # plot_cdf_list_curves([lass.copy(),lclu.copy(),lrec.copy(),\
    #                       lscc.copy(),lden.copy()],\
    #                      [r'assortativity',r'clustering',\
    #                       r'reciprocity',r'SCC',r'density'],\
    #                      'Graph metric',\
    #                      LEG_LOC='best',SET_LOG=False,FONT_SIZE_LEG=12,\
    #                      FIG_SIZE_X=6,FIG_SIZE_Y=5,\
    #                      XLIM=False,SET_GRID=False,SET_MARKER=True)    

    # -------------------------------------------------------------------------    
    
    from matplotlib.ticker import AutoMinorLocator
    minorLocatory = AutoMinorLocator(4)
    minorLocatorx = AutoMinorLocator(5)    
        
    fig, ax = plt.subplots(figsize=(7,5))
    
    ax.tick_params(axis='both',length=8, width=2,which='major',\
                    bottom=True,top=True,left=True,\
                    right=True,direction='in')
    ax.tick_params(axis='both',length=4, width=1,which='minor',\
                    bottom=True,top=True,left=True,\
                    right=True,direction='in')       
        
    ax.xaxis.set_minor_locator(minorLocatorx)
    ax.yaxis.set_minor_locator(minorLocatory)
    
    lmutual_information=np.array(lmi)
    lconditional_entropy=np.array(lhcond_d)
    
    
    plt.xlabel(r'$MI(O;D)$')
    plt.ylabel(r'$H(O|D)$')
    #plt.xlabel(r'$Entropy(O)$')
    #plt.ylabel(r'$ConditionalEntropy(O|D)$')
    # plt.scatter(lmutual_information,lconditional_entropy, marker='x', c='k')
    # plt.scatter(lmutual_information[lmutual_information>lconditional_entropy],\
    #             lconditional_entropy[lconditional_entropy<lmutual_information],\
    #                 marker='o',c='red')
    # plt.scatter(lmutual_information[lmutual_information<=lconditional_entropy],\
    #             lconditional_entropy[lconditional_entropy>=lmutual_information],\
    #                 marker='x',c='black')
    plt.hexbin(lmutual_information, lconditional_entropy, gridsize=50,cmap='Blues')
    cb = plt.colorbar() # bins='log',
    cb.set_label('Groups in bin',fontsize=18)    
    plt.tight_layout()
    plt.show()    
    # ------------------------------------------------------------------------- 

    info_mi=np.array(lmi)/np.array(lh_d)
    
    plot_cdf_list_curves([info_mi[~np.isnan(info_mi)]],\
                         ['perc. mutual info'],\
                         'Uncertainty in bits',\
                         LEG_LOC='best',SET_LOG=False,FONT_SIZE_LEG=12,\
                         FIG_SIZE_X=6,FIG_SIZE_Y=5,\
                         XLIM=False,SET_GRID=False,SET_MARKER=True)
        
    # chart_path='/media/alexandre/Seagate Expansion Drive/Whatsapp/data/plots/marginal/{0}.png'.format(fname)
    # fig.savefig(chart_path,format='png',bbox_inches='tight',dpi=300)
    # plt.clf()
    # plt.close()     
    
    # lrow_log=[]
    # for key in ['max','min']:#,'self']:
    #     values=-np.log2(dst_row[key].values.copy())
    #     lrow_log.append(values)
           
    # lcol_log=[]
    # for key in ['max','min']:
    #     values=-np.log2(dst_col[key].values.copy())
    #     lcol_log.append(values)    
        
    # plot_cdf_list_curves([lcol_log[0],lrow_log[0]],\
    #                       [r'$-\log_2~max\{Pr(O=o|D=d)$',\
    #                         # r'$-\log_2~min\{Pr(O=o|D=d)\}$',\
    #                       r'$-\log_2~max\{Pr(D=d|O=o)$'],\
    #                         # r'$-\log_2~min\{Pr(D=d|O=o)\}$'],\
    #                       'Conditional probabilities',\
    #                       LEG_LOC='best',SET_LOG=False,FONT_SIZE_LEG=12,\
    #                       FIG_SIZE_X=6,FIG_SIZE_Y=5,\
    #                       XLIM=False,SET_GRID=False,SET_MARKER=True)     
        
    # lrow_tau_log=[]
    # for key in ['max_tau','min_tau']:#,'self']:
    #     values=-np.log2(dst_row[key].values.copy()/1800.)
    #     lrow_tau_log.append(values)
           
    # lcol_tau_log=[]
    # for key in ['max_tau','min_tau']:
    #     values=-np.log2(dst_col[key].values.copy()/1800.)
    #     lcol_tau_log.append(values)          
        
    # plot_cdf_list_curves([lcol_tau_log[0],lrow_tau_log[0]],\
    #                      [r'$-\log_2~max\{Pr(O=o|D=d)$',\
    #                       # r'$-\log_2~min\{Pr(O=o|D=d)\}$',\
    #                       r'$-\log_2~max\{Pr(D=d|O=o)$'],\
    #                       # r'$-\log_2~min\{Pr(D=d|O=o)\}$'],\
    #                      'Surprisal of time fraction',\
    #                      LEG_LOC='best',SET_LOG=False,FONT_SIZE_LEG=12,\
    #                      FIG_SIZE_X=6,FIG_SIZE_Y=5,\
    #                      XLIM=False,SET_GRID=False,SET_MARKER=True)         
        
    # res=ks_2samp(-np.log2(dst_row['max'].values),\
    #              -np.log2(dst_col['max'].values))
    # res=ks_2samp(dst_row['max'].values,\
    #              dst_col['max'].values)
    # m=dst_row['max'].values.size
    # n=dst_col['max'].values.size
    # ks2=1.63*np.sqrt((m+n)/(n*m))
    # d,pvalue=res[0],res[1]
    
    # print()
    # print('KS-Test for maximum values: ')
    # print('\tKS-distance for alpha = 0.01:',ks2)
    # print('\td-value: ',d)
    # print('\tp-value: %.6f'%pvalue)
    
    # print()
    # print('Confidence interval for average maximum:')
    # avg_max_row=dst_row['max'].mean()
    # std_max_row=dst_row['max'].std()
    # bound_max_row=1.96*std_max_row/np.sqrt(dst_row['max'].values.size)
    # lower_max_row=avg_max_row-bound_max_row
    # upper_max_row=avg_max_row+bound_max_row
    # avg_max_col=dst_col['max'].mean()
    # std_max_col=dst_col['max'].std()
    # bound_max_col=1.96*std_max_col/np.sqrt(dst_col['max'].values.size)
    # lower_max_col=avg_max_col-bound_max_col
    # upper_max_col=avg_max_col+bound_max_col    
    # print('P(D=d|O=o) I.C.: (%.4f, %.4f)'%(lower_max_row,upper_max_row))
    # print('P(D=d|O=o) I.C.: (%.4f, %.4f)'%(lower_max_col,upper_max_col))
        
    # print()
    # print('Confidence interval for P(D=d|O=o) average maximum:')
    # avg_max=dst_row['max'].mean()
    # std_max=dst_row['max'].std()
    # bound_max=1.96*std_max/np.sqrt(dst_row['max'].values.size)
    # lower_max=avg_max-bound_max
    # upper_max=avg_max+bound_max
    # print('I.C. lower: ',lower_max)
    # print('I.C. upper: ',upper_max)    
    
    # # res=st.anderson_ksamp([-np.log2(dst_row['max'].values),\
    # #                    -np.log2(dst_col['max'].values)])
    
    # # res_min=ks_2samp(-np.log2(dst_row['min'].values),-np.log2(dst_col['min'].values))
    # res_min=ks_2samp(dst_row['min'].values,dst_col['min'].values)
    # m=dst_row['min'].values.size
    # n=dst_col['min'].values.size
    # ks2_min=1.63*np.sqrt((m+n)/(n*m))
    # d_min,pvalue_min=res_min[0],res_min[1]
    
    # print()
    # print('KS-Test for minimum values: ')
    # print('\tKS-distance for alpha = 0.01:',ks2_min)
    # print('\td-value: ',d_min)
    # print('\tp-value: %.6f'%pvalue_min)
    
    # print()
    # print('Confidence interval for average minimum:')
    # avg_min_row=dst_row['min'].mean()
    # std_min_row=dst_row['min'].std()
    # bound_min_row=1.96*std_max_row/np.sqrt(dst_row['min'].values.size)
    # lower_min_row=avg_min_row-bound_min_row
    # upper_min_row=avg_min_row+bound_min_row
    # avg_min_col=dst_col['min'].mean()
    # std_min_col=dst_col['min'].std()
    # bound_min_col=1.96*std_min_col/np.sqrt(dst_col['min'].values.size)
    # lower_min_col=avg_min_col-bound_min_col
    # upper_min_col=avg_min_col+bound_min_col    
    # print('P(D=d|O=o) I.C.: (%.4f, %.4f)'%(lower_min_row,upper_min_row))
    # print('P(D=d|O=o) I.C.: (%.4f, %.4f)'%(lower_min_col,upper_min_col))    
    
    # print()
    # print('Confidence interval for average minimum:')
    # avg_max=dst_row['min'].mean()
    # std_max=dst_row['min'].std()
    # bound_max=1.96*std_max/np.sqrt(dst_row['min'].values.size)
    # lower_max=avg_max-bound_max
    # upper_max=avg_max+bound_max
    # print('I.C. lower: ',lower_max)
    # print('I.C. upper: ',upper_max)    
    
    # dst_row['max']=-np.log2(dst_row['max'])
    # dst_col['max']=-np.log2(dst_col['max'])
    # dst_row['min']=-np.log2(dst_row['min'])
    # dst_col['min']=-np.log2(dst_col['min'])    
    # NUM_BINS=50
    # dst_row['max'].hist(alpha=0.5,bins=NUM_BINS,color='green')
    # dst_col['max'].hist(alpha=0.5,bins=NUM_BINS,color='blue')
