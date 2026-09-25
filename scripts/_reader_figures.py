"""Cohort coverage, synthetic NeMo explanation and paired anatomical displays."""
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Patch
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from nilearn import plotting
import numpy as np
import pandas as pd
import seaborn as sns
from _common import ROOT
from _anatomical_figures import regional_map,colourbar
from _network_context_figures import edge_heatmap


def cohort_coverage_panels(fig):
    cov=pd.read_csv(ROOT/'results/cohort_regional_coverage.csv')
    edge=pd.read_csv(ROOT/'results/cohort_edge_group_coverage.csv')
    vmax=np.ceil(100*cov.mean_chaco.max()/5)*5
    emax=np.ceil(100*edge.mean_chacoconn.max()/2)*2
    fig.text(.038,.770,'B)  Regional disconnection',fontsize=16,fontweight='bold')
    fig.text(.038,.746,'Mean ChaCo across all 163 patients',fontsize=11.5)
    regional_map(fig,(.02,.524,.50,.210),100*cov.set_index('roi_index').mean_chaco,vmax=vmax)
    colourbar(fig,[.12,.506,.29,.011],vmax,'Mean regional ChaCo (%)')
    fig.text(.60,.770,'C)  Connections by anatomical group',fontsize=16,fontweight='bold')
    fig.text(.60,.746,'All 18,145 atlas pairs; n = 163',fontsize=11.5)
    edge_heatmap(fig,[.68,.540,.255,.20],edge.assign(percent=100*edge.mean_chacoconn),
                 'percent',limit=emax)
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,emax),cmap='YlOrRd'),
                    cax=fig.add_axes([.945,.555,.010,.19]))
    bar.set_label('Mean ChaCoConn (%)',fontsize=10,fontweight='bold')
    bar.ax.tick_params(labelsize=10)


def schematic_projection(points):
    """Place illustrative nodes in a left sagittal template projection."""
    return np.asarray(points)*np.array([-65.,55.])+np.array([-15.,20.])


def schematic_background(fig,rect):
    """Use the same Nilearn anatomical outline as the study's regional maps."""
    display=plotting.plot_glass_brain(None,display_mode='l',figure=fig,axes=rect,
                                    colorbar=False,annotate=False,alpha=.20)
    ax=display.axes['l'].ax
    # Keep Nilearn's full template bounds, including the cerebellum and brainstem.
    ax.set(aspect='equal')
    return ax


def schematic_data():
    nodes=np.array([[-.80,.22],[-.48,.62],[.22,.67],[.81,.16],[.35,-.39],[-.30,-.50]])
    # Each straight edge summarises a connection, rather than depicting fibres.
    # Assigned reference/affected weights are synthetic, not inferred from a 2-D
    # line crossing. Intermediate fractions illustrate partial streamline loss.
    pairs=[(0,1),(1,2),(2,3),(3,4),(4,5),(0,3),(0,4),(1,4),(1,5)]
    reference=np.full(len(pairs),4.)
    affected=np.array([0,0,0,0,0,3,1,4,2.])
    mask=np.array([-.13,.12]);radius=.30
    matrix=np.full((6,6),np.nan)
    incident=np.zeros(6);lost=np.zeros(6)
    for (i,j),total,loss in zip(pairs,reference,affected,strict=True):
        matrix[i,j]=matrix[j,i]=100*loss/total
        incident[[i,j]]+=total;lost[[i,j]]+=loss
    return nodes,pairs,reference,affected,mask,radius,matrix,100*lost/incident,incident,lost


def connection_width(percent):
    """Positive line widths are proportional to loss; zero uses a dashed guide."""
    return .08*percent if percent>0 else 2.0


def nemo_schematic(save):
    nodes,pairs,reference,affected,mask,radius,matrix,regional,incident,lost=schematic_data()
    fig=plt.figure(figsize=(13.8,10.4))
    fig.text(.035,.965,'NeMo: regional and connection disconnection',
             fontsize=19,fontweight='bold')
    fig.text(.035,.932,'Synthetic six-region example · one straight line per connection',
             fontsize=12,color='#4c5960')
    red='#bf3f56';grey='#89949d';cmap=sns.color_palette('Blues',as_cmap=True)
    plotted_nodes=schematic_projection(nodes)
    connections=[plotted_nodes[[i,j]] for i,j in pairs]
    edge_percent=100*affected/reference
    titles=['A)  Reference network + tumour mask',
            'B)  Connection loss: ChaCoConn',
            'C)  Regional disconnection: ChaCo']
    rects=[(.03,.545,.43,.34),(.54,.545,.43,.34),(.03,.135,.43,.30)]
    for k,rect in enumerate(rects):
        ax=schematic_background(fig,rect)
        title_y=.891 if k<2 else .455
        fig.text(rect[0],title_y,titles[k],fontsize=15,fontweight='bold')
        if k==0:
            ax.add_patch(Ellipse(schematic_projection(mask),2*radius*65,2*radius*55,
                facecolor=red,edgecolor=red,alpha=.18,lw=1.4,zorder=2))
            ax.add_collection(LineCollection(connections,colors='#657681',linewidths=3.,zorder=3))
        else:
            for segment,percent in sorted(zip(connections,edge_percent,strict=True),key=lambda x:x[1]):
                ax.plot(segment[:,0],segment[:,1],color=cmap(percent/100) if percent else grey,
                        lw=connection_width(percent),linestyle='-' if percent else '--',
                        solid_capstyle='round',zorder=3)
            if k==1:
                # Labels show partial losses directly; the matrix repeats these values.
                offsets={(0,3):(.5,(3,15)),(0,4):(.60,(10,-16)),
                         (1,4):(.70,(15,0)),(1,5):(.55,(-25,0))}
                for pair,(fraction,offset) in offsets.items():
                    segment=plotted_nodes[list(pair)]
                    centre=segment[0]+fraction*(segment[1]-segment[0])
                    ax.annotate(f'{matrix[pair]:.0f}%',centre,xytext=offset,
                                textcoords='offset points',ha='center',va='center',
                                fontsize=11,fontweight='bold',zorder=5,
                                bbox={'facecolor':'white','edgecolor':'none','pad':1.2,'alpha':.94})
        for n,(x,y) in enumerate(plotted_nodes):
            colour=cmap(regional[n]/100) if k==2 else '#34434e'
            ax.scatter(x,y,s=360,color=colour,edgecolor='white' if k<2 else '#34434e',
                       linewidth=1.1,zorder=4)
            ax.text(x,y,chr(65+n),ha='center',va='center',fontsize=11,
                    fontweight='bold',color='white' if k<2 else '#1d2b35',zorder=5)
            if k==2:
                percent_label=f'{regional[n]:.1f}'.rstrip('0').rstrip('.')+'%'
                offsets=[(-18,-12),(0,18),(8,18),(22,0),(20,-14),(0,-18)]
                aligns=[('right','top'),('center','bottom'),('center','bottom'),
                        ('left','center'),('left','top'),('center','top')]
                ax.annotate(percent_label,(x,y),xytext=offsets[n],
                            textcoords='offset points',ha=aligns[n][0],va=aligns[n][1],
                            fontsize=11,fontweight='bold',zorder=5,
                            bbox={'facecolor':'white','edgecolor':'none','pad':.2,'alpha':.85})
    fig.add_artist(FancyArrowPatch((.473,.705),(.527,.705),transform=fig.transFigure,
        arrowstyle='-|>',mutation_scale=18,color='#607481',lw=1.6))
    fig.text(.035,.532,'A: Each connection has 4 reference weight units.',fontsize=11.5,color='#4c5960')
    fig.text(.54,.532,'B–C: Line width and colour show pairwise loss.',fontsize=11.5,color='#4c5960')
    handles=[Patch(facecolor=red,alpha=.23,label='Tumour mask')]
    handles += [Line2D([],[],color=cmap(p/100) if p else grey,lw=connection_width(p),
                      linestyle='-' if p else '--',label=f'{p}%') for p in [0,25,50,75,100]]
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(.5,.493),ncol=6,frameon=False,
               fontsize=11.5,columnspacing=1.5,handlelength=2.)
    fig.text(.035,.101,'Region A: (3 + 1) / 12 = 33.3%',fontsize=12)
    fig.text(.035,.074,'Affected / total reference weight at each node',fontsize=11.5,color='#4c5960')
    fig.text(.54,.455,'D)  The same connection losses as a matrix',fontsize=15,fontweight='bold')
    ax=fig.add_axes([.645,.16,.245,.27])
    sns.heatmap(matrix,mask=np.isnan(matrix),ax=ax,cmap=cmap,vmin=0,vmax=100,
        annot=True,fmt='.0f',annot_kws={'fontsize':11},square=True,linewidths=1.2,
        xticklabels=list('ABCDEF'),yticklabels=list('ABCDEF'),cbar=False)
    for label in ax.texts:
        label.set_color('white' if float(label.get_text())>60 else '#1d2b35')
    ax.tick_params(length=0,labelsize=11);ax.set_yticklabels(list('ABCDEF'),rotation=0)
    ax.set_facecolor('#e4e7ea')
    fig.text(.54,.101,'A–E: 1 / 4 = 25%     B–E: 4 / 4 = 100%',fontsize=12)
    fig.text(.54,.074,'Affected / total reference weight for each pair',fontsize=11.5,color='#4c5960')
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,100),cmap=cmap),
                    cax=fig.add_axes([.935,.195,.012,.20]))
    bar.set_label('Disconnection (%)',fontsize=11)
    bar.set_ticks([0,25,50,75,100])
    bar.ax.tick_params(labelsize=10)
    fig.text(.54,.130,'Grey cells: no reference connection',fontsize=10,color='#4c5960')
    fig.text(.035,.023,'Illustrative weights; lines join regions, not fibre paths. Study estimates use 191 parcels and healthy reference connectomes.',
             fontsize=11,color='#4c5960')
    # The manuscript places this figure at 16.8 cm; keep labels legible there.
    for label in fig.findobj(plt.Text):
        if label.get_text():
            label.set_fontsize(label.get_fontsize()+2)
    save(fig,'figure_S1_nemo_estimation')
    (ROOT/'metadata/nemo_schema.json').write_text(json.dumps({
       'type':'Synthetic explanatory diagram, not a patient or reference tractogram',
       'background':'Nilearn left sagittal glass-brain outline; same anatomical plotting system as Figure 1B',
       'example':'Six regions, nine straight connections, four reference weight units per pair; assigned affected weights illustrate partial loss',
       'regional_percent':regional.tolist(),'incident_reference_weight':incident.tolist(),
       'incident_affected_weight':lost.tolist(),'total_reference_weight':float(reference.sum()),
       'total_affected_weight':float(affected.sum()),
       'connections':[{'pair':f'{chr(65+i)}–{chr(65+j)}','nodes':[i,j],
                       'reference_weight':float(w),'affected_weight':float(d),
                       'percent':float(p),'line_width_points':connection_width(float(p))}
                      for (i,j),w,d,p in zip(pairs,reference,affected,edge_percent,strict=True)],
       'display':'One straight line per pair. Panels B and C: positive line width = 0.08 × percent; zero loss = dashed grey context line. Panel C node colour = regional ChaCo; panel D cells = pairwise ChaCoConn.',
       'geometry':'The mask is schematic. Assigned affected weights are not calculated by intersecting straight lines with this 2-D mask.',
       'pair_percent':{f'{chr(65+i)}–{chr(65+j)}':float(matrix[i,j]) for i in range(6) for j in range(i+1,6) if np.isfinite(matrix[i,j])},
       'actual_pipeline':'191 parcels; estimates averaged over healthy reference connectomes',
       'source':'https://github.com/kjamison/nemo'},indent=2)+'\n')


def paired_group_differences(fig):
    regional=pd.read_csv(ROOT/'results/high_burden_regional_differences.csv')
    edges=pd.read_csv(ROOT/'results/high_burden_edge_group_differences.csv')
    groups=pd.read_csv(ROOT/'results/high_burden_group_summary.csv')
    rv=np.ceil(regional.difference_percentage_points.abs().max()/5)*5
    ev=np.ceil(edges.difference_percentage_points.abs().max()/2)*2
    for row,battery in enumerate(['AAT','DemTect']):
        top=.972-row*.287
        d=regional[regional.battery.eq(battery)].set_index('roi_index');g=groups[groups.battery.eq(battery)].set_index('group')
        fig.text(.035,top,f'{chr(65+row*2)})  {battery}: regional difference',fontsize=16,fontweight='bold')
        fig.text(.035,top-.025,f'Deficit (n = {g.loc["Deficit","n"]}) − no deficit (n = {g.loc["No deficit","n"]})',fontsize=12)
        regional_map(fig,(.02,top-.21,.47,.177),d.difference_percentage_points,vmax=rv,signed=True)
        fig.text(.575,top,f'{chr(66+row*2)})  {battery}: connection difference',fontsize=15,fontweight='bold')
        edge_heatmap(fig,[.675,top-.211,.26,.189],edges[edges.battery.eq(battery)],'difference_percentage_points',limit=ev,signed=True)
    colourbar(fig,[.10,.403,.30,.009],rv,'Regional difference (percentage points)',signed=True)
    colourbar(fig,[.65,.380,.29,.009],ev,'Connection difference (percentage points)',signed=True)
