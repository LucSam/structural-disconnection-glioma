#!/usr/bin/env python3
"""Main and supplementary figures from complete, documented result sets."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns
from _cohort_figure import make_figure_01
from _anatomical_figures import high_burden_maps, correlation_maps, connection_maps
from _network_context_figures import patient_report, structural_context
from _reader_figures import nemo_schematic, paired_group_differences

OUT = ROOT / 'outputs/figures'
BLUE, GOLD, RED = '#2A6F97', '#BD8A35', '#8B1E3F'
STATUS = {'No deficit': BLUE, 'AAT subtest deficit only': GOLD, 'Global-score deficit': RED, 'Incomplete AAT subtests': '#999999'}
ANATOMY = {'Subcortical/limbic':'#1f77b4', 'Cerebellum':'#8c564b', 'Frontal/insula':'#d62728',
           'Perisylvian/central sulci':'#bcbd22', 'Temporal':'#9467bd', 'Parietal':'#2ca02c',
           'Occipital/visual':'#17becf', 'Cingulate/medial':'#ff7f0e'}


def theme():
    sns.set_theme(style='ticks', context='paper', font='DejaVu Sans', rc={
        'axes.labelsize':14, 'axes.labelweight':'bold', 'axes.titlesize':16,
        'axes.titleweight':'bold', 'xtick.labelsize':12, 'ytick.labelsize':12,
        'axes.linewidth':.8, 'text.color':'#151515', 'axes.labelcolor':'#151515',
        'figure.facecolor':'white', 'axes.facecolor':'white',
        'savefig.facecolor':'white', 'grid.color':'#dddddd', 'grid.linewidth':.5})


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=400, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def clean(ax, grid='y'):
    sns.despine(ax=ax)
    ax.grid(axis=grid, color='#dedede', linewidth=.55, zorder=0)
    ax.set_axisbelow(True)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight('bold')


def status(frame):
    return np.where(frame.global_below_cutoff, 'Global-score deficit',
                    np.where(frame.preserved, 'No deficit', 'AAT subtest deficit only'))


def status_legend(fig, *, bottom=.025):
    handles = [Line2D([], [], linestyle='', marker='o', markersize=7,
                      markerfacecolor=STATUS[label], markeredgecolor='white', label=label)
               for label in ['No deficit','AAT subtest deficit only','Global-score deficit']]
    handles.append(Line2D([], [], linestyle='', marker='o', markersize=9,
                          markerfacecolor='none', markeredgecolor='#111111', markeredgewidth=1.5,
                          label='High burden, no deficit'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,bottom),
               ncol=2, frameon=False, fontsize=12, handletextpad=.6, columnspacing=2.2)


def clinical_question():
    data = pd.read_csv(ROOT / 'data/private/display_participants.csv')
    thirds = pd.read_csv(ROOT / 'results/burden_tertiles.csv')
    fig = plt.figure(figsize=(12.8,14.8))
    paired_group_differences(fig)
    for i,battery in enumerate(['AAT','DemTect']):
        d=data[data.battery.eq(battery)].copy();d['status']=status(d)
        left=.085+i*.475
        ax=fig.add_axes([left,.145,.39,.165])
        x=d.burden*100
        y=d.minimum_aat_margin if battery=='AAT' else d.score
        lower,upper=np.quantile(x,[1/3,2/3]);right=x.max()*1.04
        ax.axvspan(upper,right,color='#e8ecef',alpha=.8,lw=0,zorder=0)
        for boundary in [lower,upper]:ax.axvline(boundary,color='#777777',ls=':',lw=1)
        ax.axhline(0 if battery=='AAT' else 13,color='#333333',ls='--',lw=1.3,zorder=1)
        plotted_y=y.to_numpy().copy()
        if battery=='AAT':
            tied=np.isclose(plotted_y,8)
            plotted_y[tied]+=np.random.default_rng(923).uniform(-.18,.18,tied.sum())
        sns.scatterplot(x=x,y=plotted_y,hue=d.status,palette=STATUS,s=37,alpha=.57,
                        linewidth=.35,edgecolor='white',legend=False,ax=ax,zorder=3)
        high=d.high_burden_preserved
        ax.scatter(x[high],plotted_y[high],s=85,facecolor='none',edgecolor='#111111',linewidth=1.3,zorder=4)
        ax.set_title(f'{"E" if i==0 else "F"})  {battery}: all complete cases (n = {len(d)})',loc='left',pad=15,fontsize=14.5)
        ax.set(xlim=(-.35,right),xlabel='Left-hemisphere burden (%)' if battery=='AAT' else 'Whole-brain burden (%)',
               ylabel='Lowest subtest margin\n(T-score minus its cutoff)' if battery=='AAT' else 'DemTect global score')
        if battery=='AAT':ax.set_ylim(y.min()-2,y.max()+2)
        else:ax.set(ylim=(-.6,19),yticks=[0,3,6,9,12,15,18])
        clean(ax)
        sub=thirds[thirds.battery.eq(battery)]
        fig.text(left,.084,'AAT deficit by burden third' if battery=='AAT' else 'DemTect deficit by burden third',fontsize=11,fontweight='bold')
        strip=fig.add_axes([left,.034,.39,.042],xlim=ax.get_xlim(),ylim=(0,1));strip.axis('off')
        boundaries=[0,lower,upper,right]
        for k,(_,row) in enumerate(sub.iterrows()):
            low=int(row.n-row.preserved);a,b=boundaries[k:k+2]
            strip.plot([a,a,b,b],[.82,.99,.99,.82],color='#777777',lw=.7)
            strip.text((a+b)/2,.77,f'{["Low","Middle","High"][k]}\n{low}/{int(row.n)}\n({100*low/row.n:.0f}%)',
                       ha='center',va='top',fontsize=10,linespacing=1.1)
    status_legend(fig,bottom=-.024)
    save(fig,'figure_02_disconnection_performance')


def circle_positions(meta):
    """Mirror anatomical groups on two hemisphere halves; vermis below."""
    positions, arcs = {}, []
    names=list(ANATOMY)
    for side,direction in [('L',1),('R',-1)]:
        part=meta[meta.roi_name.str.endswith(f'({side})')].copy()
        part['order']=part.anatomical_group.map({name:i for i,name in enumerate(names)})
        part['stem']=part.roi_name.str.replace(r' \([LR]\)$','',regex=True)
        part=part.sort_values(['order','stem'])
        gap=2.0;step=(168-gap*(len(names)-1))/len(part);cursor=6.0
        for group in names:
            members=part[part.anatomical_group.eq(group)]
            a=cursor;b=cursor+step*len(members)
            for k,roi in enumerate(members.roi_index):
                theta=np.deg2rad(90+direction*(a+step*(k+.5)))
                positions[int(roi)]=np.array([np.cos(theta),np.sin(theta)])
            arcs.append((side,group,90+direction*a,90+direction*b))
            cursor=b+gap
    midline=meta[meta.roi_name.str.endswith('(V)')].sort_values('roi_index')
    for roi,theta in zip(midline.roi_index,np.deg2rad(np.linspace(266,274,len(midline))),strict=True):
        positions[int(roi)]=np.array([np.cos(theta),np.sin(theta)])
    assert len(positions)==191
    return positions,arcs


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    # The cohort panel retains the original generator's font and layout settings.
    plt.rcdefaults()
    make_figure_01()
    theme()
    nemo_schematic(save)
    clinical_question();patient_report(save, clean)
    regional_groups=plt.figure(figsize=(12.8,8.0))
    # Original full regional means are retained as a supplementary figure.
    high_burden_maps(regional_groups, compact=True)
    save(regional_groups,"figure_S2_regional_group_means")
    connection_maps(save, circle_positions, ANATOMY)
    correlation_maps(save)
    structural_context(save)
    print('Created five main and five supplementary PNG figures.')
