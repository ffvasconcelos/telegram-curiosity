#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 19:46:31 2018

@author: alexandre
"""
import numpy as np
import pandas as pd
from scipy.stats import beta,kstest
import scipy.integrate as integrate
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

"""
C-value for compute the KS-value for two-tailed Kolmogorov-Smirnov test 
"""
def c_alpha(confidence):
    return np.sqrt(-(1/2.)*np.log(confidence/2.))
  
def beta_distribution_estimation(values):
    avg=np.mean(values)
    std=np.std(values)
    righ_side=(((avg*(1-avg))/std**2)-1)
    a=avg*righ_side
    b=(1-avg)*righ_side
    return a,b

def data_from_beta_dist(a,b):
    mod=(a-1)/(a+b-2)
    k=np.sqrt(((a-1)*(b-1))/(a+b-3))/(a+b-2)
    sd_max=beta.pdf(mod,a,b)
    sd_inf=-1
    sd_sup=-1
    if a > 1 and a < 2 and b > 2:
        sd_sup=beta.pdf(mod+k,a,b)
    if b > 1 and b < 2 and a > 2:
        sd_inf=beta.pdf(mod-k,a,b)
    if a > 2 and b > 2:
        sd_sup=beta.pdf(mod+k,a,b)
        sd_inf=beta.pdf(mod-k,a,b)
    return mod, k, sd_max, sd_inf, sd_sup
    
def integration_beta_dist(a,b,mode,k,minimum,maximum):
    boredom_zone=integrate.quad(beta.pdf,minimum,(mode-k),args=(a,b))[0]
    curiosity_zone=integrate.quad(beta.pdf,(mode-k),(mode+k),args=(a,b))[0]
    anxiety_zone=integrate.quad(beta.pdf,(mode+k),maximum,args=(a,b))[0]
    return boredom_zone,curiosity_zone,anxiety_zone

def beta_ks_test(values,a,b):
    #rv=np.random.beta(a,b,size=values.size)
    D,pvalue=kstest(values, 'beta',args=(a,b))
    ks_value = 1.63/float(np.sqrt(values.size))
    return D,pvalue,ks_value

def histogram(lista,size_bins,path_save_fig):
    plt.hist(lista,bins=size_bins,normed=True,alpha=0.25,facecolor='green')
    plt.draw()
    plt.savefig(path_save_fig,bbox_inches='tight')
    #plt.show()    

def plot_U_shaped(values,path_save_fig):
    values.sort()
    x_values=values.copy()
    a,b=beta_distribution_estimation(x_values)
    y_values=beta.pdf(x_values,a,b)
    sd_max=sd_inf=sd_sup=0.
    mod,k,sd_max,sd_inf,sd_sup=data_from_beta_dist(a,b)
    #print(sd_max)
    sd_max_values=np.arange(0,sd_max,0.25)
    if sd_inf > 0:
        sd_inf_values=np.arange(0,sd_inf,0.25)
    if sd_sup > 0:
        sd_sup_values=np.arange(0,sd_sup,0.25)
    xticks=np.arange(0,1,0.1)
    yticks=np.arange(0,10.1,1)
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xlim(0,1)
    plt.ylim(0,10.1)
    plt.xlabel('Stimulus Degree')
    plt.ylabel('Distribution')
    plt.grid(True,axis='both',linestyle=':')
    plt.plot([mod]*sd_max_values.size,sd_max_values,'r-.',lw=1.)
    if sd_inf > 0:
        plt.plot([mod-k]*sd_inf_values.size,sd_inf_values,'b:',lw=2.)
    if sd_sup > 0:
        plt.plot([mod+k]*sd_sup_values.size,sd_sup_values,'b:',lw=2.)
    plt.plot(x_values,y_values,'k--',lw=1.5)
    plt.hist(x_values,bins=50,normed=True,alpha=0.25,facecolor='green')
    plt.draw()
    plt.savefig(path_save_fig,bbox_inches='tight')
    #plt.show()
    plt.clf()

def mycdf(values):
    ecdf = ECDF(values)
    xvalues = pd.Series(values).unique()
    xvalues.sort()
    return xvalues, ecdf

def CDF(values, number):
    xvalues, ecdf = mycdf(values)
    plt.plot(xvalues, ecdf(xvalues))
    plt.legend()
    plt.ylabel('CDF',fontweight='bold')
    plt.xlabel(r'$x$ values',fontsize=16)
    #plt.title(r'Cluster $%d$'%number,fontweight='bold',style='italic')
    plt.show()
    plt.close()

def stats(values):
    minimum=float(np.min(values))
    maximum=float(np.max(values))
    mean=float(np.mean(values))
    std=float(np.std(values))
    return minimum,maximum,mean,std

def binary_search(values,window):
    if window <= 0:
        print("ERROR: window should be more than zero!")
        return -1
#    print(values)
    left=0
    right=len(values)-1
    index=-1
    while left <= right:
        midle=int((left+right)/2)
        refer = values[len(values)-1] - values[midle] #+ 1
#        print('midle:',midle,'refer:',refer)
        if refer == window:
            return midle
#            break
        if refer < window:
            right = midle - 1
#            print('left:',left,'right:',right)
#            print()
        if refer > window:
            left = midle + 1
#            print('left:',left,'right:',right)
#            print()
        #if left == right:
        index = left
        #    break

#    print('Index: ',index)
#    print()
    return index

def plot_cdf_list_curves(list_curve,list_label,x_label,chart_path='',\
                         SET_LOG=True,CCDF=False,LEG_LOC='',XLIM=False,xlimits=[],YLIM=False,ylimits=[],\
                         OUT_LEG=False,SAVE_FIG=False,SET_TITLE=False,title_name='',\
                         FONT_SIZE_LEG=17,FONT_SIZE=17,FIG_SIZE_X=7,FIG_SIZE_Y=5,BOUND_CORR=False,\
                             SET_MARKER=False,SET_GRID=False,SET_LEG=True):
    #import matplotlib
    from matplotlib.ticker import AutoMinorLocator,FormatStrFormatter#,MultipleLocator
    #matplotlib.use("Agg")
    #import matplotlib.pyplot as plt
    from statsmodels.distributions.empirical_distribution import ECDF
    fig, ax =plt.subplots(figsize=(FIG_SIZE_X,FIG_SIZE_Y))
    
    size = len(list_curve)
    #style = ['-','-.','--',':','-','--','-.',':','-.','--','-','-.']
    style = [':','-','--',':','-','--','-.',':','-.','--','-','-.']
    #marker = ['-o','-.^','--D',':s',':v','-.+','--D','-*','-.','-.d','--h',':>']
    #color = ['blue','black','red','green','purple','brown','magenta','pink','goldenrod','cyan','grey','yellow']
    color = ['black','red','black','green','purple']
    # color = ['black','black','black','black','black','black','black','black','black', 'black','black','black']
    linewidth = [2.0,2.5,2.,2.8,3.8,3.5,3.,2.5,2.5,3.,3.5,3.8]
    minorLocatory = AutoMinorLocator(4)
    minorLocatorx = AutoMinorLocator(4)
    
    # And a corresponding grid
    #ax.grid(which='both')

    # Or if you want different settings for the grids:
    #ax.grid(which='minor', alpha=0.2)
    #ax.grid(which='major', alpha=0.5)
    
    #major_ticks = np.arange(0, 1.01, 0.2)
    #minor_ticks = np.arange(0, 1.01, 0.1)
    #ax.set_yticks(major_ticks)
    #ax.set_yticks(minor_ticks, minor=True)    
    for i in range(size):
        print(i,size)
        list_curve[i].sort()
        #list_curve_2.sort()
        ecdf = ECDF(list_curve[i])
        #ecdf_curve_2 = ECDF(list_curve_2)
        if SET_LOG:
            #ax.set_yscale('log')
            ax.set_xscale('log')
        

        ax.tick_params(axis='both',length=9, width=2.2,which='major',bottom=True,top=True,left=True,\
                       right=True,direction='in',labelsize=FONT_SIZE)
        ax.tick_params(axis='both',length=6, width=1.5,which='minor',bottom=True,top=True,left=True,\
                       right=True,direction='in',labelsize=FONT_SIZE)
        ax.yaxis.set_minor_locator(minorLocatory)
        if not SET_LOG:
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        markers_on=[]
        for idx in range(34):
            idx_value=np.int(np.ceil(idx*0.03*len(list_curve[i])))
            markers_on.append(idx_value)
        markers_on.append(len(list_curve[i])-1)
        if SET_GRID:
            plt.grid(True)
        if not SET_LOG:
            ax.xaxis.set_minor_locator(minorLocatorx)
        if XLIM==True:
            ax.set_xlim(xlimits)
        if YLIM==True:
            ax.set_ylim(ylimits)
        if CCDF and BOUND_CORR != True:
            if not SET_MARKER:
                plt.plot(ecdf.x, 1 - ecdf.y, label=list_label[i],lw=linewidth[i],\
                     ls=style[i],c=color[i])
            else:
                # plt.plot(ecdf.x, 1 - ecdf.y, label=list_label[i],lw=linewidth[i],\
                # ls=style[i],c=color[i],marker=marker[i],markevery=markers_on)
                plt.plot(ecdf.x,1-ecdf.y, label=list_label[i],\
                         c=color[i], ms=4,lw=linewidth[i],ls=style[i])#, marker[i],markevery=markers_on)
                #plt.plot(ecdf.x, 1 - ecdf.y, label=list_label[i])
        else:
            if not SET_MARKER:
                plt.plot(ecdf.x, ecdf.y, label=list_label[i],lw=linewidth[i],\
                         ls=style[i],c=color[i])
            else:
                # plt.plot(ecdf.x, ecdf.y, label=list_label[i],lw=linewidth[i],\
                # ls=style[i],c=color[i],marker=marker[i],markevery=markers_on)
                plt.plot(ecdf.x, ecdf.y, label=list_label[i],\
                         c=color[i], ms=4,lw=linewidth[i],ls=style[i])#, marker[i],markevery=markers_on)
            #plt.plot(ecdf.x, ecdf.y, label=list_label[i])
        if BOUND_CORR == True and size == 3:
            plt.fill([-0.5,-0.5,0.5,0.5],[0,1,1,0],color='red',alpha=0.005)
   
    #plt.grid(linestyle='--',linewidth=1.)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.legend(loc='upper left')
    #plt.legend(loc='lower left')
    plt.rcParams.update({'font.size': FONT_SIZE_LEG, 'font.family': 'sans-serif',\
                         'axes.linewidth':'3.'})  
    if SET_TITLE:
        ax.set_title(title_name)
    ax.set_xlabel(x_label,fontsize=FONT_SIZE)#,fontweight='bold')
    if CCDF:
        if LEG_LOC=='' and OUT_LEG == True:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        else:
            plt.legend(loc=LEG_LOC)
        ax.set_ylabel('CCDF',fontsize=FONT_SIZE+2)#,fontweight='bold')
    else:
        if LEG_LOC=='' and OUT_LEG == True:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        else:
            plt.legend(loc=LEG_LOC)
        #ax.set_ylabel('CDF',fontsize=FONT_SIZE+2)#,fontweight='bold')
        ax.set_ylabel(r'$P(X \leq x)$',fontsize=FONT_SIZE+2)#,fontweight='bold')
    if not SET_LEG:
        ax.legend().set_visible(False)
    #plt.figure(figsize=(FIG_SIZE_X,FIG_SIZE_Y))
    plt.tight_layout()
    if SAVE_FIG:
        plt.draw()
        fig.savefig(chart_path,format='pdf',bbox_inches='tight',dpi=300)
        plt.clf()
        plt.close()
    else:
        plt.show()
    #plt.close()
    #plt.clf()
    
"""
Boxplot Function
"""
def plot_boxplot_list(list_data,label,OUTLIERS=False,SET_TITLE=False,\
                      title_chart='',SET_LOG=False,SET_XLABEL=False,\
                      xlabel='',SET_YLABEL=False,ylabel=''):
    from matplotlib.ticker import AutoMinorLocator
    minorLocatory = AutoMinorLocator(2)

    fig,ax=plt.subplots()
    ax.set_title(title_chart)
    #ax.set_ylim([0.8,12])
    ax.yaxis.grid(True)
    ax.tick_params(length=6, width=2, which='major',bottom=True,top=True,left=True,right=True,direction='in')
    ax.tick_params(axis='both',which='minor',direction='in',left=True,right=True,top=True)
    # ax.grid(which='minor',alpha=0.8,ls='--')
    ax.grid(axis='y',which='major',alpha=1.,ls='-.')
    ax.yaxis.set_minor_locator(minorLocatory)
    if SET_LOG == True:
        ax.set_yscale('log')
    if SET_YLABEL == True:
        ax.set_ylabel(ylabel,fontsize=17)
    medianprops=dict(linewidth=2.,color='black')
    meanpointprops = dict(marker='D',markeredgecolor='black',markerfacecolor='red')
    bp0=ax.boxplot(list_data[:2],showfliers=OUTLIERS,patch_artist=False,vert=True,\
               meanprops=meanpointprops,medianprops=medianprops,meanline=False,\
                   showmeans=True,positions=[1,2],widths = 0.4)
    
    bp1=ax.boxplot(list_data[2:],showfliers=OUTLIERS,patch_artist=False,vert=True,\
               meanprops=meanpointprops,medianprops=medianprops,meanline=False,\
                   showmeans=True,positions=[3,4],widths = 0.4)
    
    plt.setp(bp0['boxes'][1],color='blue')
    plt.setp(bp0['whiskers'][2],color='blue')
    plt.setp(bp0['whiskers'][3],color='blue')
    plt.setp(bp0['caps'][2],color='blue')
    plt.setp(bp0['caps'][3],color='blue')    
    plt.setp(bp0['medians'][1],color='blue')
    bp0['fliers'][0].set_color('b')
    bp0['fliers'][1].set_color('b')
    plt.setp(bp0['fliers'][0])
    plt.setp(bp0['fliers'][1])
    
    plt.setp(bp1['boxes'][1],color='blue')
    plt.setp(bp1['whiskers'][2],color='blue')
    plt.setp(bp1['whiskers'][3],color='blue')
    plt.setp(bp1['caps'][2],color='blue')
    plt.setp(bp1['caps'][3],color='blue')    
    plt.setp(bp1['medians'][1],color='blue')
    plt.setp(bp1['fliers'][0],color='blue')
    plt.setp(bp1['fliers'][1],color='blue')
    
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='k', lw=1.2),
                    Line2D([0], [0], color='b', lw=1.2)]    
    plt.legend(custom_lines,['Conditional Entropy','Mutual Information'])
    # number=np.arange(len(label)+1)
    # limit=len(list_data)+1
    plt.xticks([1.5,3.5],label)#(number[1:limit],label[:limit])
    if SET_XLABEL == True:
        plt.xlabel(xlabel)
    plt.show()
