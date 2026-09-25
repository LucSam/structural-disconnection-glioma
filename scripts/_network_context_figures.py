"""Anatomical context and performance/disconnection/QoL displays."""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns

from _common import ROOT, OUTCOMES
from _anatomical_figures import regional_map, colourbar

GROUPS=['Frontal/insula','Perisylvian/central sulci','Temporal','Parietal',
        'Occipital/visual','Cingulate/medial','Subcortical/limbic','Cerebellum']
SHORT=['Frontal/insula','Perisylvian','Temporal','Parietal','Occipital','Cingulate','Subcortical','Cerebellum']
STATUS={'No deficit':'#2A6F97','AAT subtest deficit only':'#BD8A35',
        'Global-score deficit':'#8B1E3F','Incomplete test data':'#999999'}
SCALES=[('EORTC_QLQ_C30_global_health','Global health\n/ QoL'),
        ('EORTC_QLQ_C30_cognitive_functioning','Cognitive\nfunctioning'),
        ('EORTC_QLQ_BN20_communication_deficit','Communication')]


def group_matrix(frame,value):
    matrix=np.zeros((8,8))
    for r in frame.itertuples():
        a,b=GROUPS.index(r.group_i),GROUPS.index(r.group_j)
        matrix[a,b]=matrix[b,a]=getattr(r,value)
    return matrix


def edge_heatmap(fig,rect,frame,value,*,limit,signed=False,labels=True):
    ax=fig.add_axes(rect)
    matrix=group_matrix(frame,value)
    sns.heatmap(matrix,ax=ax,cmap='RdBu_r' if signed else 'YlOrRd',
                vmin=-limit if signed else 0,vmax=limit,cbar=False,square=True,
                linewidths=.35,linecolor='white',
                xticklabels=SHORT,yticklabels=SHORT if labels else False)
    ax.set(xlabel='',ylabel='')
    ax.tick_params(axis='both',length=0,pad=3,labelsize=11.5)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45,ha='right')
    ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
    return ax


def structural_context(save):
    correlations=pd.read_csv(ROOT/'results/regional_display_correlations.csv')
    differences=pd.read_csv(ROOT/'results/high_burden_edge_group_differences.csv')
    means=pd.read_csv(ROOT/'results/high_burden_edge_group_means.csv')
    groups=pd.read_csv(ROOT/'results/high_burden_group_summary.csv')
    dmax=np.ceil(differences.difference_percentage_points.abs().max()/2)*2
    fig=plt.figure(figsize=(12.5,6.8))
    for k,(name,d) in enumerate(correlations[correlations.scope.eq('global')].groupby('outcome',sort=False)):
        top=.945-k*.405
        fig.text(.035,top,f'{chr(65+k)})  {name} (n = {d.n.iloc[0]})',fontsize=17,fontweight='bold')
        regional_map(fig,(.025,top-.325,.95,.295),d.set_index('roi_index').deficit_rho,
                     vmax=.30,signed=True,mode='lyrz')
    colourbar(fig,[.31,.063,.38,.019],.30,'Partial deficit correlation (ρ)',signed=True)
    save(fig,'figure_03_regional_associations')

    # Unselected group means accompany the contrasts, including zero values.
    fig=plt.figure(figsize=(15.6,11.2))
    meanmax=np.ceil(100*means.mean_chacoconn.max()/5)*5
    for row,battery in enumerate(['AAT','DemTect']):
        top=.95-row*.435
        g=groups[groups.battery.eq(battery)].set_index('group')
        for col,group in enumerate(['No deficit','Deficit','Difference']):
            left=.105+col*.31
            if group=='Difference':
                d=differences[differences.battery.eq(battery)];value='difference_percentage_points'
                label='Deficit − no deficit';lim=dmax
            else:
                d=means[means.battery.eq(battery)&means.group.eq(group)].copy()
                d['percentage']=100*d.mean_chacoconn;value='percentage'
                label=f'{group} (n = {g.loc[group,"n"]})';lim=meanmax
            fig.text(left-.04,top,f'{chr(65+row*3+col)})  {battery}: {label}',fontsize=13.5,fontweight='bold')
            edge_heatmap(fig,[left,top-.29,.25,.27],d,value,limit=lim,signed=group=='Difference')
    colourbar(fig,[.21,.045,.29,.012],meanmax,'Group mean ChaCoConn (%)')
    colourbar(fig,[.72,.045,.22,.012],dmax,'Deficit − no deficit (pp)',signed=True)
    save(fig,'figure_S3_group_connection_disconnection')


def patient_report(save,clean):
    data=pd.read_csv(ROOT/'data/private/network_qol_participants.csv')
    classified=pd.read_csv(ROOT/'data/private/display_participants.csv')
    corr=pd.read_csv(ROOT/'results/qol_correlations.csv')
    structural=pd.read_csv(ROOT/'results/network_qol_correlations.csv')
    measures=[('Mean AAT T-score','AAT_mean_T_score'),*OUTCOMES['AAT'].items(),
              ('DemTect global','DemTect_global'),*OUTCOMES['DemTect'].items()]
    qcols=[v for v,_ in SCALES]
    matrix=corr.pivot(index='neuropsych_measure',columns='qol_measure',values='spearman_rho').loc[
        [c for _,c in measures],qcols]
    qs=corr.pivot(index='neuropsych_measure',columns='qol_measure',values='q_fdr_all_neuropsych_qol_tests').loc[matrix.index,matrix.columns]
    annotations=np.array([[f'{matrix.iloc[i,j]:.2f}'+('*' if qs.iloc[i,j]<.05 else '')
                           for j in range(3)] for i in range(len(measures))])
    fig=plt.figure(figsize=(16.2,13.3))
    fig.text(.035,.963,'A)  Test performance and reported function',fontsize=16,fontweight='bold')
    ax=fig.add_axes([.203,.421,.253,.510])
    sns.heatmap(matrix,ax=ax,cmap='RdBu_r',vmin=-.6,vmax=.6,annot=annotations,fmt='',
                annot_kws={'fontsize':13},linewidths=.75,cbar=False)
    ax.set(xlabel='',ylabel='',xticklabels=[],yticklabels=[name for name,_ in measures])
    ax.tick_params(axis='y',rotation=0,labelsize=12,length=0,pad=7)
    ax.tick_params(axis='x',length=0);ax.axhline(5,color='#555555',lw=1.3)
    fig.text(.203,.404,'* FDR q <0.05 across 222 tests',fontsize=11)
    fig.text(.035,.365,'B)  Disconnection and reported function',fontsize=16,fontweight='bold')
    features=['left_mean','whole_mean','AAT_edges','DemTect_edges']
    smatrix=structural.pivot(index='feature',columns='qol_measure',values='rho').loc[features,qcols]
    ax=fig.add_axes([.203,.155,.253,.185])
    sns.heatmap(smatrix,ax=ax,cmap='RdBu_r',vmin=-.6,vmax=.6,annot=True,fmt='.2f',
                annot_kws={'fontsize':13},linewidths=.75,cbar=False)
    ax.set(xlabel='',ylabel='',xticklabels=[label for _,label in SCALES],
           yticklabels=['Left-hemisphere burden','Whole-brain burden','ChaCoConn: AAT edges','ChaCoConn: DemTect edges'])
    ax.tick_params(axis='y',rotation=0,labelsize=12,length=0,pad=7)
    ax.tick_params(axis='x',rotation=0,labelsize=10.5,length=0,pad=7)
    ax.axhline(2,color='#555555',lw=1.3)
    fig.text(.203,.105,'Descriptive ρ; n = 64 / 62 / 70 by column',fontsize=10.5)
    colourbar(fig,[.24,.067,.18,.011],.6,'Spearman ρ',signed=True)

    perf_smoothers=pd.read_csv(ROOT/'results/qol_display_smoothers.csv')
    q='EORTC_QLQ_BN20_communication_deficit'
    for col,(battery,score,burden) in enumerate([('AAT','AAT_mean_T_score','left_mean'),('DemTect','DemTect_global','whole_mean')]):
        flags=classified[classified.battery.eq(battery)][['subject_id','high_burden_preserved','preserved','global_below_cutoff']]
        d=data.merge(flags,on='subject_id',how='left',validate='one_to_one')
        d['status']=np.where(d[score]<(63.5 if battery=='AAT' else 13),'Global-score deficit',
            np.where(d.preserved.eq(True),'No deficit',np.where(d.preserved.eq(False),
            'AAT subtest deficit only','Incomplete test data')))
        for row,key in enumerate([score,burden]):
            left, bottom=.535+col*.25,.647-row*.414
            ax=fig.add_axes([left,bottom,.208,.265])
            sub=d.dropna(subset=[key,q]).copy();rng=np.random.default_rng(310+row*2+col)
            x=sub[key].to_numpy()*(100 if row else 1)
            y=sub[q].to_numpy()+rng.uniform(-.5,.5,len(sub))
            if row==0:x=x+rng.uniform(-.1,.1,len(sub))
            sns.scatterplot(x=x,y=y,hue=sub.status,palette=STATUS,s=34,alpha=.64,
                            linewidth=.3,edgecolor='white',legend=False,ax=ax,zorder=3)
            high=sub.high_burden_preserved.eq(True).to_numpy()
            ax.scatter(x[high],y[high],s=80,facecolor='none',edgecolor='#111111',linewidth=1.3,zorder=4)
            if row==0:
                smooth=perf_smoothers[perf_smoothers.battery.eq(battery)]
                sx,sy=smooth.score,smooth.fitted_communication
                result=corr[corr.neuropsych_measure.eq(score)&corr.qol_measure.eq(q)].iloc[0]
                annotation=f'ρ = {result.spearman_rho:.2f}; q = {result.q_fdr_all_neuropsych_qol_tests:.3f}; n = {int(result.n)}'
                title=f'{"C" if col==0 else "D"})  {battery} performance'
                xlabel='Mean AAT T-score' if col==0 else 'DemTect global score'
            else:
                result=structural[structural.feature.eq(burden)&structural.qol_measure.eq(q)].iloc[0]
                annotation=f'ρ = {result.rho:.2f}; n = {int(result.n)} (descriptive)'
                title=f'{"E" if col==0 else "F"})  {"Left-hemisphere" if col==0 else "Whole-brain"} burden'
                xlabel='Disconnection burden (%)'
            if row==0:
                curve,=ax.plot(sx,sy,color='#555555',lw=1.4,zorder=2)
                curve.set_clip_path(Rectangle((sx.min(),0),sx.max()-sx.min(),100,transform=ax.transData))
            ax.set_title(title,loc='left',fontsize=14.5,pad=33)
            ax.text(0,1.037,annotation,transform=ax.transAxes,fontsize=10.7)
            ax.set(ylim=(-4,105),yticks=[0,25,50,75,100],xlabel=xlabel,
                   ylabel='Communication score' if col==0 else '')
            ax.xaxis.label.set_size(12);ax.yaxis.label.set_size(12)
            if row==0 and col==0:ax.set(xlim=(49,80),xticks=[50,60,70,80])
            if row==0 and col==1:ax.set(xlim=(-.5,18.6),xticks=[0,6,12,18])
            clean(ax)
    fig.text(.535,.171,'Colours follow AAT groups in C/E and DemTect groups in D/F.',fontsize=10.5)
    fig.text(.535,.151,'Structural plots include patients with missing test data (grey).',fontsize=10.5)
    handles=[Line2D([],[],linestyle='',marker='o',markersize=7,markerfacecolor=v,
                    markeredgecolor='white',label=k) for k,v in STATUS.items()]
    handles.append(Line2D([],[],linestyle='',marker='o',markersize=9,markerfacecolor='none',
                          markeredgecolor='#111111',markeredgewidth=1.3,label='High burden, no deficit'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.76,.010),ncol=2,frameon=False,fontsize=10.2)
    save(fig,'figure_05_patient_reported_function')
