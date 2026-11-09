import sys
import pandas as pd
import numpy as np
import scipy.optimize as opt
import scipy.stats as sts
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs

from collections import defaultdict
from typing import Tuple
import os


from omniCNV import Model as Model
from omniCNV import Solution as Solution

clonecolor = defaultdict (lambda: 'gray')
clonecolor.update({'A' : 'tab:blue',
                   'LAB' : 'tab:cyan',
                   'AB' : 'tab:green', 
                   'AABB' : 'black',
                   'AAAB' : 'tab:purple',
                   'AA' : 'tab:brown',
                   'AAB' : 'tab:red', 
                   'AAA' : 'tab:orange',
                   'CRAZY' : 'tab:pink',
                   '-1' : 'gray'})


def _reserve_figure_title_space(fig, top_axes=0.86):
    """Shrink the subplot area so suptitle/subtitle sit above axes (constrained_layout)."""
    try:
        le = fig.get_layout_engine()
        if le is not None and hasattr(le, 'set'):
            le.set(rect=(0.02, 0.02, 0.98, top_axes))
            return
    except Exception:
        pass
    try:
        fig.subplots_adjust(left=0.06, right=0.99, bottom=0.05, top=top_axes)
    except Exception:
        pass


def _distance_for_plot_title(genome):
    """Scalar d for the figure subtitle: prefer Solution.get_distance(), then Genome.distance."""
    sol = genome.segments
    d = None
    if hasattr(sol, 'get_distance'):
        d = sol.get_distance()
    elif hasattr(sol, 'distance'):
        d = getattr(sol, 'distance')
    else:
        d = getattr(genome, 'distance', float('nan'))
    if d is None:
        return float('nan')
    if isinstance(d, pd.Series):
        d = d.iloc[0] if len(d) else float('nan')
    try:
        return float(d)
    except (TypeError, ValueError):
        return float('nan')


class Genome:
    """
    Integrates allele-count and gene-count segmentations,
    runs karyotyping optimisation, and produces output reports.
    """
    def __init__ (self, gene_counts, allele_counts, coverage_type, config, logger, sample_name = None):
        """Combine gene-count and allele-count segmentations.
        Performs karyotyping.

        The input objects must provide ``get_segments()`` and compatible
        segment tables.  Merged segments are created during initialization and
        stored in ``self.segments``.
        """
        self.config = config
        self.logger = logger.getChild (f'{self.__class__.__name__}')
        self.gene_counts = gene_counts
        self.allele_counts = allele_counts
        self.coverage_type = coverage_type
        self.sample_name = sample_name
        self.chromsizes = pd.read_csv (self.config['FILES']['cyto'], sep = '\t')

        self.gene_segments = self.gene_counts.get_segments ()
        self.allele_segments = self.allele_counts.get_segments ()

        self.merge_segments ()


    def intersect_segments (self, ai_seg, cv_seg):
        """Intersect allele-imbalance and coverage intervals.

        The two inputs are per-arm DataFrames containing ``start``, ``end``,
        ``m``, and ``s``.  Returns a DataFrame with aligned intervals and
        ``ai``, ``dai``, ``cv``, and ``dcv`` values.
        """
        
        if ai_seg.empty or cv_seg.empty:
            return pd.DataFrame(columns=['start', 'end', 'ai', 'dai', 'cv', 'dcv'])

        #Aligne edges
        A2, B2 = self._align_edges(ai_seg, cv_seg)
        
            # Two-pointer sweep (O(n+m))
        res_rows = []
        i, j = 0, 0
        # Access as numpy for speed
        a_starts = A2['start'].to_numpy()
        a_ends   = A2['end'].to_numpy()
        a_mvals  = A2['m'].to_numpy()
        a_svals  = A2['s'].to_numpy()

        b_starts = B2['start'].to_numpy()
        b_ends   = B2['end'].to_numpy()
        b_mvals  = B2['m'].to_numpy()
        b_svals  = B2['s'].to_numpy()

        na = len(A2)
        nb = len(B2)

        while i < na and j < nb:
            a_s, a_e = a_starts[i], a_ends[i]
            b_s, b_e = b_starts[j], b_ends[j]

            start = a_s if a_s >= b_s else b_s
            end   = a_e if a_e <= b_e else b_e

            if (start <= end):
                res_rows.append({'start': start,
                                 'end': end,
                                 'ai': a_mvals[i],
                                 'dai': a_svals[i],
                                 'cv': b_mvals[j],
                                 'dcv': b_svals[j],})

            # Advance the interval that ends first; if tie, advance both
            if a_e < b_e:
                i += 1
            elif a_e > b_e:
                j += 1
            else:
                i += 1
                j += 1

        return pd.DataFrame(res_rows, columns=['start', 'end', 'ai', 'dai', 'cv', 'dcv'])

    
    def _align_edges(self, A: pd.DataFrame, B: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extend only the first and last intervals so both lists share the same global start and end.
        Returns copies; does not mutate the inputs.
        """
        if A.empty or B.empty:
            return A.copy(), B.copy()
        A2 = A.sort_values(by='start').copy()
        B2 = B.sort_values(by='start').copy()
        L = min(A2.iloc[0]['start'], B2.iloc[0]['start'])
        R = max(A2.iloc[-1]['end'],   B2.iloc[-1]['end'])
        A2.at[A2.index[0], 'start'] = L
        B2.at[B2.index[0], 'start'] = L
        A2.at[A2.index[-1], 'end'] = R
        B2.at[B2.index[-1], 'end'] = R
        return A2, B2


    def merge_segments (self):
        """Merge overlapping allele and coverage segments across chromosome arms.

        Populates ``self.segments`` with complete merged intervals and
        ``self.na_segments`` with intervals whose signal is missing.
        """
        arms = list (set.intersection (set(self.gene_segments.keys()), set(self.allele_segments.keys())))
        segments = []
        for arm in arms:
            self.logger.debug (f'Merging segments for {arm}')
            merged_segs = self.intersect_segments (self.allele_segments[arm], self.gene_segments[arm])
            merged_segs['arm'] = arm
            self.logger.info (f'Finished merging segments for {arm}. Total segments: {len(merged_segs)}')
            segments.append (merged_segs)

        all_segments = pd.concat(segments, ignore_index = True)  
        all_segments['size'] = 1e-6 * (all_segments['end'] - all_segments['start'])
        self.logger.info (f'Total merged segments: {len(all_segments)}. Total size: {all_segments["size"].sum():.2f} Mb')

        self.segments = all_segments.loc[(~all_segments['cv'].isna())&(~all_segments['ai'].isna())].copy()
        
        tmp = all_segments.loc[all_segments['cv'].isna() | all_segments['ai'].isna(), ]
        if len(tmp):
            self.logger.error (f'Found {len(tmp)} segments with missing values. They are removed.')
            self.logger.error (f'{tmp}')
        self.na_segments = tmp.copy()
        #self.merge_segments = self.segments.copy()


    def generate_report (self, sample_name, report_name):
        """
        Dispatch to the appropriate report-generation method.

        Parameters
        ----------
        sample_name : str  Path prefix for output files.
        report_name : str  One of 'plot_clones', 'ambiguous_regions'.
        """
        if report_name in ['plot_clones']:
            distances = np.asarray(
                [solution.distance for solution in self.solutions], dtype=float
            )
            # np.argsort sends NaN to the end; replace NaN so ordering is stable for multi-solution runs
            distances_sort = np.where(np.isnan(distances), np.inf, distances)
            solution_indexes = np.argsort(distances_sort, kind='mergesort')    
            
            for i,solution in enumerate([self.solutions[idx] for idx in solution_indexes]):
                self.segments = solution
                self.m0 = solution.m0
                self.scale = solution.scale
                self.distance = solution.distance
                if report_name == 'plot_clones':
                    self.plot_clones(sample_name+ f'_solution_{i+1}')
                #elif report_name == 'igv':
                    tmp = self.segments.get_segments()[['arm', 'start', 'end', 'ai', 'cn']].copy()
                    tmp['ID'] = 'UNPOP'
                    tmp['chr'] = [a[:-1] for a in tmp['arm'].tolist()]
                    tmp = tmp[['ID','chr', 'start', 'end', 'ai', 'cn']]
                    tmp.set_index (["ID", 'chr', 'start', 'end'], inplace = True) 
                    report = tmp.stack (level = -1, future_stack = True).reset_index()
                    report.columns = ["ID", "chr", "start", "end", "probe", "data"]
                    report.sort_values (by = ["ID","chr", "start", "probe"], inplace = True)
                    
                    with open (sample_name + f'_solution_{i+1}_cn' + '.seg', 'w') as f:
                        f.writelines ("#type=COPY_NUMBER\n")
                        report['ID'] = sample_name.split('/')[-1] + '_cn'
                        report.loc[report["probe"] == 'cn'].to_csv(f, sep = '\t', index = False)
                    
                    with open (sample_name + f'_solution_{i+1}_ai' + '.seg', 'w') as f:
                        f.writelines ("#type=LOH\n")
                        report['ID'] = sample_name.split('/')[-1] + '_ai'
                        report.loc[report["probe"] == 'ai'].to_csv(f, sep = '\t', index = False)

                    with open (sample_name + f'_solution_{i+1}_gen_segs.tsv', 'w') as f:
                        self.segments.get_segments()[['arm','start','end','ai','dai','cv','dcv','size','zygosity','cn','dcn','k','good']].to_csv (f, sep = '\t', index = False)
        elif report_name in ['ambiguous_regions']:
            results = []
            for arm, armseg in self.segments.get_segments().groupby (by = 'arm'):
                starts = armseg['start'].values
                ends = armseg['end'].values
                segsizes = ends - starts
                ambsizes = starts[1:] - ends[:-1]
                                    
                margins = np.concat ((segsizes[:-1][:,np.newaxis]/2,
                                      ambsizes[:,np.newaxis]/2, 
                                      segsizes[1:][:,np.newaxis]/2), axis = 1).min (axis = 1)
                

                results.append (pd.DataFrame.from_dict({'chrom': np.repeat (arm[:-1], len(starts)-1),
                                                        'start': (ends[:-1] - margins).astype(int),
                                                        'end': (starts[1:] + margins).astype(int)}))
            pd.concat (results).to_csv (sample_name+"_ambiguous_regions.bed", sep  = '\t', index = False)
            
        else:
            self.logger.error (f'Unknown report type: {report_name}')
    
    
    def karyotype (self, models):
        """Performs karyotyping based on the provided models.

        ``models`` identifies the allowed clone models.  The method partitions
        segments by balance and size, runs the configured optimization, and
        stores candidate :class:`Solution.Solution` objects in
        ``self.solutions``.
        """
        
        self.models = Model.Model (models_labels = models,
                                   config = self.config,
                                   logger = self.logger)
        self.segments['zygosity'] = '-1'
        
        balance_filter = self.segments['ai'] < float(self.config['KARYOTYPING']['max_dip_ai'])
        size_filter = self.segments['size'] >= float(self.config['KARYOTYPING']['MIN_SIZE'])

        self.balanced_segments = self.segments.loc[balance_filter & size_filter, ].copy()
        self.imbalanced_segments = self.segments.loc[~balance_filter & size_filter, ].copy()
        self.balanced_small_segments = self.segments.loc[balance_filter & ~size_filter, ].copy()
        self.imbalanced_small_segments = self.segments.loc[~balance_filter & ~size_filter, ].copy()

        if (self.allele_counts.chrY is not None):
            if (len(self.allele_counts.chrY.segments) > 0):
                self.allele_counts.chrY.data['l2r'] = self.allele_counts.chrY.data['cv']/self.gene_counts.median_cv
            
                self.allele_counts.chrY.segments['cv'] = self.allele_counts.chrY.segments['cv']/self.gene_counts.median_cv
                self.allele_counts.chrY.segments['dcv'] = self.allele_counts.chrY.segments['dcv']/self.gene_counts.median_cv
            
                self.imbalanced_small_segments = pd.concat ([self.imbalanced_small_segments, self.allele_counts.chrY.segments],
                                                             ignore_index = True)
                self.allele_counts.data = pd.concat ([self.allele_counts.data, self.allele_counts.chrY.data],
                                                      ignore_index = True)
                self.gene_counts.data = pd.concat ([self.gene_counts.data, self.allele_counts.chrY.data],
                                                    ignore_index = True)
                self.imbalanced_small_segments['zygosity'] = '-1'
                self.imbalanced_segments['size'] = (self.imbalanced_segments['end'] - self.imbalanced_segments['start'])/10**6

        total_size = np.sum((self.balanced_segments['size'].sum(),
                             self.imbalanced_segments['size'].sum(), 
                             self.balanced_small_segments['size'].sum(),
                             self.imbalanced_small_segments['size'].sum()))

        ##TODO: Test for any nans in segments 
        genome_size = np.sum (self.chromsizes.groupby (by = 'chrom')['chromEnd'].max())/10**6 
        self.logger.info (f'Total segments size: {total_size:.2f} Mb')
        self.logger.info (f'Segments cover: {100*total_size/genome_size:.2f} % of genome')
        self.logger.info (f'Balanced segments are ({100*self.balanced_segments["size"].sum()/total_size:.2f} %)')
        self.logger.info (f'Imbalanced segments are ({100*self.imbalanced_segments["size"].sum()/total_size:.2f} %)')
        self.logger.info (f'Small balanced segments are ({100*self.balanced_small_segments["size"].sum()/total_size:.2f} %)')
        self.logger.info (f'Small imbalanced segments are ({100*self.imbalanced_small_segments["size"].sum()/total_size:.2f} %)')

        if (100*self.balanced_segments["size"].sum()/total_size) < 0.01:
            self.logger.critical ('Less than 1% of genome is balanced. Karyotyping is (currently) impossible.')
            self.logger.info ('Top most ballanced segments: ')
            self.logger.info (self.imbalanced_segments[['arm', 'start', 'end', 'ai', 'cv']].sort_values (by = 'ai', ascending = True).head(10))
            if self.sample_name is not None:
                pd.concat([self.balanced_segments,
                           self.imbalanced_segments,
                           self.balanced_small_segments,
                           self.imbalanced_small_segments]).to_csv (self.sample_name + '_no_solution_gen_segs.tsv',
                                                                    sep = '\t', index = False)
                self.allele_counts.data.to_csv (self.sample_name + '_no_solution_allele_counts.tsv', sep = '\t', index = False)
            sys.exit (1)
        
        self.m0range = self.models.get_m0_range(self.balanced_segments,
                                                self.imbalanced_segments,
                                                0.25/np.sqrt(self.gene_counts.median_cv))
        
        
        self.solutions = []
        for m0range in self.m0range:
            mmin, minran, maxran, mmax = m0range
            self.logger.info (f'Examining m0 range: {m0range}')
            balanced = self.balanced_segments.copy()
            balanced['zygosity'] = 'AB'
            balanced.loc[balanced['cv'] < mmin, 'zygosity'] = 'LAB'
            balanced.loc[balanced['cv'] > mmax, 'zygosity'] = 'AABB'

            balanced_small = self.balanced_small_segments.copy()
            balanced_small['zygosity'] = 'AB'
            balanced_small.loc[balanced_small['cv'] < mmin, 'zygosity'] = 'LAB'
            balanced_small.loc[balanced_small['cv'] > mmax, 'zygosity'] = 'AABB'

            self.solutions.append (Solution.Solution (mbounds = (minran, maxran),
                                                      init_scale = float(self.config['KARYOTYPING']['initial_scale']),
                                                      balanced = balanced,
                                                      balanced_small = balanced_small,
                                                      imbalanced = self.imbalanced_segments.copy(),
                                                      imbalanced_small = self.imbalanced_small_segments.copy(),
                                                      coverage_type = self.coverage_type,
                                                      models = self.models,
                                                      config = self.config,
                                                      logger = self.logger))

    
    def plot_clones (self, sample_name):
        """Save a multi-panel clone and segment visualization as a PNG."""
        segments = self.segments.get_segments().copy()
        armsizes = pd.read_csv (self.config['FILES']['arms'], sep = '\t')
        armsizes['arm'] = armsizes['Name']
        chromsizes = self.chromsizes
        max_chrom = np.nanmax([int(c[3:]) for c in chromsizes['chrom'] if c.startswith('chr') and c[3:].isdigit()])

        fg = plt.figure(figsize = (20,12), constrained_layout = True)
        # Same padding (inches) for horizontal and vertical gaps so the space between
        # axai–axmain matches axmain–axcn (left vs bottom marginals to main).
        _cl_pad = 0.02
        try:
            fg.set_constrained_layout_pads(w_pad=_cl_pad, h_pad=_cl_pad)
        except Exception:
            pass
        # width_ratios: 2 cols (axai) + 6 (axmain) + 12 (right subfigure) = 20
        gs = mgs.GridSpec(
            12,
            20,
            figure=fg,
            width_ratios=[0.9, 0.9] + [1.0] * 6 + [1.0] * 12,
        )
        axmain = fg.add_subplot(gs[0:6, 2:8])
        axai = fg.add_subplot(gs[0:6, 0:2], sharey = axmain)
        axcn = fg.add_subplot(gs[6:8, 2:8], sharex = axmain)
        axlegend = fg.add_subplot(gs[6:8, 0:2])

        ##TODO Move to config?
        chromorder = ['chr' + str(c) for c in range(1, max_chrom+1)] + ['chrX', 'chrY']
        sbfg = fg.add_subfigure(gs[0:8, 8:20])
        # Use gridspec_kw instead of subplots_adjust (incompatible with parent constrained_layout).
        axschrs = sbfg.subplots(
            4,
            6,
            sharex = True,
            sharey = True,
            gridspec_kw = dict(hspace = 0, wspace = 0,
                               left = 0.11, bottom = 0.11,
                               top = 0.99),
        )

        axaichr = fg.add_subplot(gs[10:, :])
        axcovchr = fg.add_subplot(gs[8:10, :])

        self.plot_karyotypes(axmain, axai, axcn, axlegend)
        self.plot_karyotypes_chrom(axschrs, chromorder)
        self.plot_per_chromosome(axaichr, axcovchr, chromorder, chromsizes)

        _reserve_figure_title_space(fg, top_axes = 0.86)
        _d = _distance_for_plot_title(self)
        fg.suptitle(f'{os.path.basename(sample_name)}', y = 0.978)
        fg.text(
            0.5,
            0.948,
            f'd: {_d:.2f}, m0: {self.m0:.2f}, scale: {self.scale:.2f}, coverage: {self.gene_counts.median_cv:.2f}', 
            ha='center',
            va='top',
            color='grey',
            fontsize=10,
            transform=fg.transFigure,
        )

        fg.savefig(sample_name + '_clones.png', dpi=150, metadata={})
        plt.close(fg)

    
    def plot_karyotypes (self, axmain, axai, axcn, axlegend):
        """Draw the main allelic-imbalance/copy-number karyotype plot.

        Out-of-range copy numbers are clipped for display and marked with
        directional markers; the input segment table is not modified.
        """
        
        k = np.linspace (0,1,20)
        axmain.plot ((1-k/2)*2, k/(2*(2-k)), color = clonecolor['A'], linestyle = ':')
        axmain.plot ((1+k/2)*2, k/(2*(2+k)), color = clonecolor['AAB'], linestyle = ':')
        axmain.plot (np.repeat (2, len(k)), k/2, color = clonecolor['AA'], linestyle = ':')
        axmain.plot (2+k, 3*k/(4+2*k), color = clonecolor['AAA'], linestyle = ':')
        k = np.linspace (1,2,20)
        axmain.plot ((1+k/2)*2, k/(2*(2+k)), color = clonecolor['AAAB'], linestyle = ':')

        axmain.text (2, 0, "AB", fontsize = 12, fontweight = 'bold', va = 'top', 
                     ha = 'center', color = clonecolor['AB'])
        axmain.text (3, 1/6, "AAB", fontsize = 12, fontweight = 'bold', va = 'bottom',
                      ha = 'center', color = clonecolor['AAB'])
        axmain.text (3, 1/2, "AAA", fontsize = 12, fontweight = 'bold', va = 'bottom',
                      ha = 'center', color = clonecolor['AAA'])
        axmain.text (4, 1/4, "AAAB", fontsize = 12, fontweight = 'bold', va = 'bottom',
                      ha = 'center', color = clonecolor['AAAB'])
        axmain.text (2, 1/2, "AA", fontsize = 12, fontweight = 'bold', va = 'bottom',
                      ha = 'center', color = clonecolor['AA'])
        axmain.text (1, 1/2, "A", fontsize = 12, fontweight = 'bold', va = 'bottom',
                      ha = 'center', color = clonecolor['A'])

        segments = self.segments.get_segments().copy()
        segments['marker'] = 'o'
        
        filt = segments['cn'].astype(float) < 0.5
        segments.loc[filt, 'cn'] = 0.6
        segments.loc[filt, 'marker'] = '<'
        segments.loc[filt, 'zygosity'] = 'CRAZY'
        filt = segments['cn'].astype(float) > 4.5
        segments.loc[filt, 'cn'] = 4.4
        segments.loc[filt, 'marker'] = '>'
        segments.loc[filt, 'zygosity'] = 'CRAZY'
        
        for marker, segs in segments.groupby('marker'):
            x = segs['cn'].to_numpy()
            y = segs['ai'].to_numpy()
            c = [clonecolor[m] for m in segs['zygosity']]
            s = segs['size'].to_numpy(dtype = float)
            axmain.scatter(x, y, c=c, marker=marker, edgecolors='none', s=s, linewidths=0)

        ais = np.linspace (-0.01, 0.5, 100)
        cns = np.linspace (segments['cn'].min(), segments['cn'].max(), 100)  
        for model, modeldata in segments.loc[segments['size'] > 10].groupby('zygosity'):
            try:
                tmp = sts.gaussian_kde(modeldata['ai'].values)(ais)
                axai.plot (tmp/tmp.max(), ais, 
                           color = clonecolor[model]) 
            except:
                pass
            try:
                tmp = sts.gaussian_kde(modeldata['cn'].values)(cns)
                axcn.plot (cns, tmp/tmp.max(), 
                           color = clonecolor[model])
            except:
                pass

        axlegend.axis ('off')

        axmain.set_xlabel ('')
        axai.set_xlabel ('pdf')
        axai.set_ylabel ('allelic imbalance (ai)')
        axcn.set_xlabel ('copy number (cn)')
        axcn.set_ylabel ('pdf')

        axcn.set_yticks ([])
        axai.set_xticks ([])
        # Marginal x is normalized pdf in [0, 1]; default x-margins leave empty band on the
        # right that reads as a gap before the main scatter - pin x to [0, 1] with no margin.
        axai.margins (x=0)
        axai.set_xlim (0, 1)

        cnrange = axcn.get_xlim()
        axcn.set_xlim ((np.max((cnrange[0], 0.5)), np.min((cnrange[1], 4.5))))
        xt = np.arange (0.5, 5, 0.5)
        axcn.set_xticks (xt)
        axcn.set_xticklabels (xt.astype(str))
        # Shared x: show numeric tick labels only on the bottom PDF marginal (axcn), not on axmain.
        axmain.tick_params (axis = 'x', labelbottom = False)
        # Shared y (allelic imbalance): show tick labels only on the left PDF marginal (axai), not on axmain.
        axmain.tick_params (axis = 'y', labelleft = False)


    def plot_per_chromosome (self, axaichr, axcovchr, chromorder, chromsizes):
        """Draw allele-frequency and coverage tracks along the linearized genome."""
        start = 0
        xTicksLab = []
        xTicksPos = []
        segments = self.segments.get_segments().copy()
        
        for chrom in chromorder:
            axaichr.plot ((start, start), (0,1), 'k:', lw = 1)
            axcovchr.plot ((start, start), (-1,3), 'k:', lw = 1)
            for arm in [chrom + 'p', chrom + 'q']:
                if len (segments.loc[segments['arm'] == arm]):
                    xTicksLab.append(arm)
                    tmp = self.allele_counts.data.loc[(self.allele_counts.data['arm'] == arm)]
                    xTicksPos.append (0.5*(tmp['Pos'].min() + tmp['Pos'].max()) + start)
                for _, seg in segments.loc[segments['arm'] == arm].iterrows():
                    tmpai = self.allele_counts.data.loc[(self.allele_counts.data['arm'] == arm)&\
                                                        (self.allele_counts.data['cv'] >= float(self.config['AC_FILTER']['cvthr']))&\
                                                        (self.allele_counts.data['Pos'] >= seg['start'])&\
                                                        (self.allele_counts.data['Pos'] <= seg['end']), ]
                    tmpcov = self.gene_counts.data.loc[(self.gene_counts.data['arm'] == arm)&\
                                                       (self.gene_counts.data['Pos'] >= seg['start'])&\
                                                       (self.gene_counts.data['Pos'] <= seg['end']), ]
                    color = clonecolor[seg['zygosity']]
                    axaichr.plot (tmpai['Pos'].values + start,
                                  tmpai['vaf'].values,
                                  color = color, ms = 0.5,
                                  marker = '.', lw = 0,
                                  alpha = 0.5 if self.coverage_type == 'wgs' else 1)
                    axaichr.plot ((seg['start']+start, seg['end']+start),
                                  (0.5-seg['ai'], 0.5-seg['ai']), 'k-')
                    axaichr.plot ((seg['start']+start, seg['end']+start),
                                  (0.5+seg['ai'], 0.5+seg['ai']), 'k-')
                    axcovchr.plot (tmpcov['Pos'].values + start,
                                   tmpcov['l2r'].values,
                                   color = color, ms = 0.5,
                                   marker = '.', lw = 0, alpha = 0.5 if self.coverage_type == 'wgs' else 1)
                    
                    axcovchr.plot ((seg['start']+start, seg['end']+start), (seg['cv'], seg['cv']), 'k-')
                    
            start +=  chromsizes.loc[chromsizes['chrom'] == chrom, 'chromEnd'].max()

        axaichr.set_xticks (xTicksPos)
        axaichr.set_xticklabels (xTicksLab, rotation = 90, ha = 'center')
        axcovchr.set_xticks ([],[])
        #segments.loc[segments['arm'] == chromorder[0]+'p','start'].min() - 10**5
        axcovchr.set_xlim ((-10**4, start))
        axaichr.set_xlim (axcovchr.get_xlim())
        
        cvmin = []
        cvmax = []
        
        for arm in self.gene_counts.data['arm'].unique():
            a,b,c = self.gene_counts.data.loc[self.gene_counts.data['arm'] == arm, 'l2r'].quantile(q=[0.5-0.159,0.5,0.5+0.159])
            cvmin.append (b - 5* (c - a))
            cvmax.append (b + 5* (c - a))

        axcovchr.set_ylim (np.min(cvmin), np.max(cvmax))
        axaichr.set_ylabel ('vaf')
        axcovchr.set_ylabel ('relative coverage')
        axaichr.set_xlabel ('linearized genomic position')


    def plot_karyotypes_chrom (self, axschrs, chromorder):
        """Draw one karyotype panel per chromosome.
        """
        segments = self.segments.get_segments().copy()
        segments['marker'] = 'o'
        
        filt = segments['cn'].astype(float) < 0.5
        segments.loc[filt, 'cn'] = 0.6
        segments.loc[filt, 'marker'] = '<'
        segments.loc[filt, 'zygosity'] = 'CRAZY'
        filt = segments['cn'].astype(float) > 4.5
        segments.loc[filt, 'cn'] = 4.4
        segments.loc[filt, 'marker'] = '>'
        segments.loc[filt, 'zygosity'] = 'CRAZY'
        segments['size'] = 1e-6 * (segments['end'] - segments['start'])
        
        for chrom, ax in zip (chromorder, axschrs.flatten()):
            for marker,segs in segments.loc[segments['arm'].str.startswith(chrom)].groupby(by = 'marker'):
                segments.plot.scatter (x = 'cn', y = 'ai', ax = ax, color = 'tab:gray',
                                       marker = marker, alpha = 0.3,
                                       s = segments['size'].values)
            tmp = segments.loc[[c[:-1] == chrom for c in segments['arm'].tolist()]]
            ax.text (3.5, 0.45, chrom, fontsize = 8, fontweight = 'bold', va = 'top',
                     ha = 'center', color = 'black')
            xmin = np.min ((segments['cn'].min(), 0.5))
            xmax = np.max ((segments['cn'].max(), 4.5))

            for y in [0, 0.5-1/3, 0.5-1/4]:
                ax.plot ((xmin, xmax),(y,y), '--', color = 'gray', alpha = 1)
            for x in [0.5*2, 2, 1.5*2, 2*2]:
                ax.plot ((x, x),(0,0.5), '--', color = 'gray', alpha = 1)
            
            for _,t in tmp.iterrows():
                ax.text (t['cn'], t['ai'], t['arm'][-1], fontsize = 10, fontweight = 'bold',
                         va = 'center', ha = 'center', color = clonecolor[t['zygosity']])