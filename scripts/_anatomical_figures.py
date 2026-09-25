"""Atlas-filled regional maps and uncapped, hemisphere-ordered edge displays."""
import json

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Arc, Patch
from matplotlib.collections import LineCollection
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting

from _common import ROOT, anatomical_group


def regional_map(fig, rect, values, *, vmax, signed=False, mode='lzr'):
    atlas = nib.load(ROOT / 'data/resources/nemo_fs191_parcellation.nii.gz')
    labels = np.asarray(atlas.dataobj, dtype=int)
    lookup = np.zeros(int(labels.max()) + 1)
    for roi, value in values.items():
        lookup[int(roi)] = float(value)
    image = nib.Nifti1Image(lookup[labels], atlas.affine)
    return plotting.plot_glass_brain(
        image, display_mode=mode, figure=fig, axes=rect, colorbar=False,
        cmap='RdBu_r' if signed else 'YlOrRd', symmetric_cbar=signed,
        vmax=vmax, vmin=-vmax if signed else 0, plot_abs=False,
        threshold=1e-12, black_bg=False, annotate=True)


def colourbar(fig, rect, vmax, label, *, signed=False):
    ax = fig.add_axes(rect)
    norm = Normalize(-vmax if signed else 0, vmax)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='RdBu_r' if signed else 'YlOrRd'),
                      cax=ax, orientation='horizontal')
    bar.set_label(label, fontsize=10, fontweight='bold')
    bar.ax.tick_params(labelsize=10)


def high_burden_maps(fig, compact=False):
    means = pd.read_csv(ROOT / 'results/high_burden_parcel_means.csv')
    differences = pd.read_csv(ROOT / 'results/high_burden_regional_differences.csv')
    summary = pd.read_csv(ROOT / 'results/high_burden_group_summary.csv')
    vmax = np.ceil(100 * means.mean_chaco.max() / 5) * 5
    dmax = np.ceil(differences.difference_percentage_points.abs().max() / 5) * 5
    for row, battery in enumerate(['AAT', 'DemTect']):
        top = .945 - row * (.44 if compact else .23)
        fig.text(.04, top, f'{"A" if row == 0 else "B"})  {battery}: highest burden third',
                 fontsize=17, fontweight='bold')
        for col, group in enumerate(['No deficit', 'Deficit', 'Difference']):
            left = .04 + col * .325
            if col < 2:
                d = means[means.battery.eq(battery) & means.group.eq(group)].set_index('roi_index')
                r = summary[summary.battery.eq(battery) & summary.group.eq(group)].iloc[0]
                title = f'{group} (n = {int(r.n)})'
                subtitle = f'Mean burden: {100*r.mean_burden:.1f}%'
                values = 100 * d.mean_chaco
            else:
                d = differences[differences.battery.eq(battery)].set_index('roi_index')
                title, subtitle = 'Deficit − no deficit', 'Difference in percentage points'
                values = d.difference_percentage_points
            fig.text(left, top-(.05 if compact else .030), title, fontsize=12.5, fontweight='bold')
            fig.text(left, top-(.083 if compact else .051), subtitle, fontsize=10.5)
            regional_map(fig, (left-.007, top-(.35 if compact else .198), .304, .25 if compact else .143), values,
                         vmax=dmax if col == 2 else vmax, signed=col == 2)
    colourbar(fig, [.14, .065 if compact else .503, .38, .016 if compact else .009], vmax, 'Mean regional ChaCo (%)')
    colourbar(fig, [.715, .065 if compact else .503, .22, .016 if compact else .009], dmax, 'Deficit − no deficit (pp)', signed=True)


def correlation_maps(save):
    """Supplementary maps for the nine principal subtests."""
    data = pd.read_csv(ROOT / 'results/regional_display_correlations.csv')
    subtests = data[data.scope.eq('subtest')]
    fig = plt.figure(figsize=(13, 12.2))
    for i, (name, d) in enumerate(subtests.groupby('outcome', sort=False)):
        row, col = divmod(i, 3)
        top, left = .955-row*.295, .025+col*.33
        fig.text(left, top, f'{chr(65+i)})  {name}', fontsize=13, fontweight='bold')
        fig.text(left, top-.025, f'{d.battery.iloc[0]} · n = {d.n.iloc[0]}', fontsize=11)
        regional_map(fig, (left-.006, top-.242, .313, .207), d.set_index('roi_index').deficit_rho,
                     vmax=.45, signed=True, mode='lzr')
    colourbar(fig, [.32, .045, .36, .012], .45, 'Partial deficit correlation (ρ)', signed=True)
    save(fig, 'figure_S4_subtest_regional_associations')


def connection_curve(p, q):
    """Draw one independent cubic arc between two parcel endpoints."""
    t = np.linspace(0, 1, 49)[:, None]
    return (1-t)**3*p + 3*(1-t)**2*t*(.42*p) + 3*(1-t)*t*t*(.42*q) + t**3*q


def connection_maps(save, circle_positions, anatomy):
    main = pd.read_csv(ROOT / 'results/tfnbs_hemisphere/edge_statistics.csv.gz')
    original = pd.read_csv(ROOT / 'data/reference/tfnbs_edges.csv')
    lobe = original[original.model_family.eq('lobe')]
    counts = pd.read_csv(ROOT / 'results/connection_model_comparison.csv')
    meta = pd.read_csv(ROOT / 'data/resources/network_definitions.csv').sort_values('roi_index')
    meta['anatomical_group'] = meta.roi_name.map(anatomical_group)
    meta['side'] = meta.roi_name.str.extract(r'\(([LRV])\)$')
    positions, arcs = circle_positions(meta)
    tmax = np.ceil(max(main.loc[main.significant_battery_bonferroni, 't_value'].abs().max(),
                       lobe.loc[lobe.significant_fwe_edges, 't_value'].abs().max()))
    norm = Normalize(0, tmax)
    cmap = plt.colormaps['magma_r']
    drawn = []
    for sensitivity, data, threshold, stem in [
        (False, main, 'significant_battery_bonferroni', 'figure_04_connection_anatomy'),
        (True, lobe, 'significant_battery_bonferroni', 'figure_S5_lobe_connection_anatomy')]:
        shown = counts[counts.lobe_across_subtests.gt(0) if sensitivity else counts.hemisphere_across_subtests.gt(0)]
        columns = 2 if sensitivity else 3
        fig = plt.figure(figsize=(12, 10.5 if not sensitivity else 6.6))
        h = fig.get_figheight()
        panel_h = 4.55 if not sensitivity else 5.25
        for k, r in enumerate(shown.itertuples()):
            row, col = divmod(k, columns)
            top, left = .963-row*panel_h/h, .025+col*.965/columns
            ax = fig.add_axes([left, top-(panel_h-.15)/h, .93/columns, (panel_h-.68)/h])
            ax.set(xlim=(-1.14,1.14), ylim=(-1.2,1.23), aspect='equal'); ax.axis('off')
            selected = data[data.subtest_label.eq(r.subtest) & data[threshold]].copy()
            selected['strength'] = selected.t_value.abs()
            selected = selected.sort_values('strength')
            segments, colors, widths = [], [], []
            for edge in selected.itertuples():
                i,j = int(edge.roi_i),int(edge.roi_j)
                segments.append(connection_curve(positions[i], positions[j]))
                fraction = float(norm(edge.strength))
                color = list(cmap(fraction))
                color[3] = .22 + .16*fraction
                colors.append(color)
                widths.append(.28+.55*fraction)
            ax.add_collection(LineCollection(segments, colors=colors, linewidths=widths, zorder=1))
            active = set(selected.roi_i)|set(selected.roi_j)
            for side, group, start, end in arcs:
                ax.add_patch(Arc((0,0),2.10,2.10,theta1=min(start,end),theta2=max(start,end),
                                 color=anatomy[group],lw=5,zorder=3))
            for node in meta.itertuples():
                p = positions[int(node.roi_index)]
                ax.scatter(*p,s=11 if node.roi_index in active else 4,color=anatomy[node.anatomical_group],
                           edgecolor='white',linewidth=.2,alpha=1 if node.roi_index in active else .4,zorder=4)
            ax.text(-.53,1.17,'Left',ha='center',fontsize=12,fontweight='bold')
            ax.text(.53,1.17,'Right',ha='center',fontsize=12,fontweight='bold')
            ax.text(0,-1.18,'Vermis',ha='center',fontsize=9)
            fig.text(left,top,f'{chr(65+k)})  {r.subtest}',fontsize=15,fontweight='bold')
            label = f'{len(selected):,} connections across subtests'
            fig.text(left,top-.25/h,label,fontsize=11.5)
            drawn.append({'figure':stem,'subtest':r.subtest,'criterion':threshold,'n_drawn':len(segments),
                          'n_qualifying':len(selected),'cap':None,'routing':'independent cubic arcs; no bundling'})
        fig.legend(handles=[Patch(color=v,label=k) for k,v in anatomy.items()],loc='lower center',
                   bbox_to_anchor=(.5,.062),ncol=4,frameon=False,fontsize=10.5,columnspacing=1.2)
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),
                          cax=fig.add_axes([.30,.037,.4,.012]),orientation='horizontal')
        bar.set_label('Edge-wise t statistic',fontsize=10)
        bar.ax.tick_params(labelsize=9)
        save(fig,stem)
    (ROOT / 'metadata/connection_figure_counts.json').write_text(json.dumps(drawn,indent=2)+'\n')
