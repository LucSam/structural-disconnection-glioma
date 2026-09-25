#!/usr/bin/env python3
"""Alternative TFNBS display with anatomical sectors and parcel connection counts.

All qualifying main-model edges are displayed. Counts describe this thresholded
association graph, not hubs or degree in the healthy reference connectome.
"""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/.mplconfig'))
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Wedge, Patch
import numpy as np
import pandas as pd
import seaborn as sns
from _common import anatomical_group

OUT = ROOT / 'outputs/alternatives'
GROUPS = {
    'Subcortical/limbic': ('Subcort.', '#1f77b4'),
    'Cerebellum': ('Cerebellum', '#8c564b'),
    'Frontal/insula': ('Frontal/insula', '#d62728'),
    'Perisylvian/central sulci': ('Perisylvian', '#bcbd22'),
    'Temporal': ('Temporal', '#9467bd'),
    'Parietal': ('Parietal', '#2ca02c'),
    'Occipital/visual': ('Occipital', '#17becf'),
    'Cingulate/medial': ('Cingulate', '#ff7f0e'),
}


def layout(meta, union):
    degree = pd.concat([union.roi_i, union.roi_j]).value_counts()
    meta = meta.copy()
    meta['degree_order'] = meta.roi_index.map(degree).fillna(0)
    angles, widths, arcs = {}, {}, []
    for side, direction in [('L', 1), ('R', -1)]:
        part = meta[meta.roi_name.str.endswith(f'({side})')]
        gap = 2.7
        step = (164 - gap*(len(GROUPS)-1))/len(part)
        cursor = 8.
        for group in GROUPS:
            members = part[part.anatomical_group.eq(group)].sort_values(
                ['degree_order', 'roi_index'], ascending=[False, True])
            start, end = cursor, cursor+len(members)*step
            for k, roi in enumerate(members.roi_index):
                angles[int(roi)] = np.deg2rad(90+direction*(start+step*(k+.5)))
                widths[int(roi)] = np.deg2rad(step*.82)
            arcs.append((side, group, 90+direction*start, 90+direction*end))
            cursor = end+gap
    midline = meta[meta.roi_name.str.endswith('(V)')].sort_values('roi_index')
    for roi, angle in zip(midline.roi_index, np.linspace(266,274,len(midline)), strict=True):
        angles[int(roi)] = np.deg2rad(angle)
        widths[int(roi)] = np.deg2rad(.8)
    assert len(angles) == 191
    return angles, widths, arcs


def panel(ax, selected, meta, angles, widths, arcs, norm, cmap, max_degree):
    ax.set(aspect='equal', xlim=(-1.40,1.40), ylim=(-1.43,1.51))
    ax.axis('off')
    degree = pd.concat([selected.roi_i, selected.roi_j]).value_counts()
    segments, colours, line_widths = [], [], []
    t = np.linspace(0,1,65)[:,None]
    # Independent smooth chords, without hierarchical bundles or edge selection.
    for r in selected.assign(strength=selected.t_value.abs()).sort_values('strength').itertuples():
        p = np.array([np.cos(angles[r.roi_i]),np.sin(angles[r.roi_i])])
        q = np.array([np.cos(angles[r.roi_j]),np.sin(angles[r.roi_j])])
        c = .48
        segments.append((1-t)**3*p+3*(1-t)**2*t*c*p+3*(1-t)*t*t*c*q+t**3*q)
        strength = float(norm(r.strength))
        colours.append(cmap(strength));line_widths.append(.38)
    ax.add_collection(LineCollection(segments,colors=colours,linewidths=line_widths,zorder=1))
    for side, group, start, end in arcs:
        lo,hi=min(start,end),max(start,end)
        ax.add_patch(Wedge((0,0),1.066,lo,hi,width=.046,
                          facecolor=GROUPS[group][1],edgecolor='white',linewidth=.2,zorder=3))
        # Faint constant-radius count guide across each anatomical sector.
        ax.add_patch(Wedge((0,0),1.35,lo,hi,width=.0015,facecolor='#dce1e5',zorder=0))
    for node in meta.itertuples():
        roi=int(node.roi_index);angle=angles[roi]
        count=int(degree.get(roi,0));height=.27*count/max_degree
        ax.add_patch(Wedge((0,0),1.08+height,np.rad2deg(angle-widths[roi]/2),
                    np.rad2deg(angle+widths[roi]/2),width=max(height,.001),
                    facecolor=GROUPS[node.anatomical_group][1],edgecolor='none',zorder=3))
        if count:
            ax.plot(np.cos(angle),np.sin(angle),'.',color=GROUPS[node.anatomical_group][1],
                    markersize=2,zorder=4)
    ax.text(-.69,1.42,'Left',ha='center',fontsize=10.5,fontweight='bold')
    ax.text(.69,1.42,'Right',ha='center',fontsize=10.5,fontweight='bold')
    ax.text(0,-1.39,'Vermis',ha='center',fontsize=8.5)
    assert int(degree.sum()) == 2*len(selected)
    return len(segments),degree


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    source=ROOT/'results/tfnbs_hemisphere/edge_statistics.csv.gz'
    data=pd.read_csv(source)
    selected=data[data.significant_battery_bonferroni].copy()
    counts=pd.read_csv(ROOT/'results/connection_model_comparison.csv')
    shown=counts[counts.hemisphere_across_subtests.gt(0)]
    meta=pd.read_csv(ROOT/'data/resources/network_definitions.csv').sort_values('roi_index')
    meta['anatomical_group']=meta.roi_name.map(anatomical_group)
    union=selected[['roi_i','roi_j']].drop_duplicates()
    angles,widths,arcs=layout(meta,union)
    # Only the positive deficit-direction contrast is displayed. Use its observed
    # range rounded outwards, and exactly the same colour scale in all panels.
    assert selected.t_value.gt(0).all()
    norm=Normalize(np.floor(selected.t_value.min()),np.ceil(selected.t_value.max()))
    cmap=sns.color_palette('magma',as_cmap=True)
    max_degree=int(np.ceil(max(pd.concat([g.roi_i,g.roi_j]).value_counts().max()
                              for _,g in selected.groupby('subtest_label'))/20)*20)
    fig=plt.figure(figsize=(13.2,17.3))
    gs=fig.add_gridspec(3,2,left=.025,right=.975,top=.968,bottom=.175,hspace=.17,wspace=.04)
    record=[];nodes=[]
    for k,r in enumerate(shown.itertuples()):
        d=selected[selected.subtest_label.eq(r.subtest)]
        ax=fig.add_subplot(gs[k//2,k%2])
        drawn,degree=panel(ax,d,meta,angles,widths,arcs,norm,cmap,max_degree)
        assert drawn==r.hemisphere_across_subtests
        ax.set_title(f'{chr(65+k)})  {r.subtest}  ·  {drawn:,} connections',loc='left',
                     fontsize=13.5,fontweight='bold',pad=15)
        record.append({'subtest':r.subtest,'n_drawn':drawn,'n_qualifying':int(r.hemisphere_across_subtests),
                       'degree_sum':int(degree.sum()),'cap':None})
        for n in meta.itertuples():
            nodes.append({'subtest':r.subtest,'roi_index':n.roi_index,'roi_name':n.roi_name,
                          'anatomical_group':n.anatomical_group,'displayed_connection_count':int(degree.get(n.roi_index,0)),
                          'angle_radians':float(angles[n.roi_index])})
    fig.legend(handles=[Patch(facecolor=colour,label=group) for group,(_,colour) in GROUPS.items()],
               loc='upper center',bbox_to_anchor=(.5,.164),ncol=4,frameon=False,
               fontsize=10.5,columnspacing=1.5,handlelength=1.2)
    cbar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),
                     cax=fig.add_axes([.30,.090,.40,.010]),orientation='horizontal')
    cbar.set_label('Connection–performance association (t statistic)',fontsize=11)
    fig.text(.5,.044,'Higher t: stronger evidence of greater disconnection with lower test performance.',
             ha='center',fontsize=10.5)
    fig.text(.5,.025,f'Outer bars: displayed connections per region (common scale 0–{max_degree}).',
             ha='center',fontsize=11)
    fig.savefig(OUT/'figure_04_connection_anatomy_alternative.png',dpi=350,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    pd.DataFrame(nodes).to_csv(OUT/'figure_04_alternative_node_counts.csv',index=False)
    meta_out={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'criterion':'Existing main-model edge-FWE and battery-wise Bonferroni correction',
        'order':'Hemisphere, anatomical group, decreasing incident-edge count in the unique union of all six displayed subtests; same order in every panel',
        'bar_scale':[0,max_degree], 'bar_interpretation':'Incident displayed association edges, not reference-connectome hubs',
        'routing':'Independent cubic chords; all retained edges; no bundling',
        'colour':'Magma; positive edge-wise t statistic, common across panels; opaque lines of equal width',
        'colour_limits':[float(norm.vmin),float(norm.vmax)],
        'colour_interpretation':'Greater disconnection associated with lower performance; t is not disconnection burden or an effect-size estimate',
        'inspiration':'Bassett & Sporns (2017), Network neuroscience, Figure 7; original rendering of current study results',
        'source_url':'https://pmc.ncbi.nlm.nih.gov/articles/PMC5485642/',
        'panels':record}
    (ROOT/'metadata/alternative_circle_counts.json').write_text(json.dumps(meta_out,indent=2)+'\n')
    print('Alternative exported; all displayed counts verified:',[(r['subtest'],r['n_drawn']) for r in record])


if __name__=='__main__':
    main()
