import numpy as np
import pandas as pd
import sys
import scipy.optimize as opt
import scipy.stats as sts

import matplotlib.pyplot as plt

from omniCNV import Segments as Segments

TMM_Acut = 0.05
TMM_Mcut = 0.3


class GeneCounts:
    """
    Reads and handles gene count data, calculates ratios, and finds segments.
    Provides methods to access data and generate reports.
    """
    def __init__(self, data_file, reference_panel, data_type, config, logger,
                 skip_segmentation = False, apply_log_to_coverage = False):
        """Load gene counts, calculate normalized coverage, and segment it.

        ``data_file`` may be a path or a DataFrame.  File input must contain
        the gene and count columns named by the configuration; the reference
        panel is required for non-WGS data.  The resulting segmented data is
        available through :meth:`get_data` and :meth:`get_segments`.
        """
        self.config = config
        self.logger = logger.getChild(f'{self.__class__.__name__}')

        if isinstance(data_file, pd.DataFrame):
            self.logger.info(f'Loading {data_type} data from DataFrame')
            self.data = data_file
        else:
            self.logger.info(f'Loading {data_type} data from {data_file}')
            self.fetch_genecounts(data_file, reference_panel)
        
        self.msnfun = msnfun #if data_type != 'wgs' else msnfun_wgs
            
        self.calculate_ratios (data_type, apply_log_to_coverage = apply_log_to_coverage)
        self.logger.info (f"Median coverage: {self.median_cv}")
        self.find_segments(skip_segmentation = skip_segmentation)
        

    def fetch_genecounts (self, file, reference_panel) -> None:
        """Read sample counts and the reference panel into DataFrames.

        The configured input columns are normalized to ``geneID`` and
        ``count``.  For the reference panel, excluded chromosome arms are
        removed according to the configuration.
        """
        try:
            tmp = pd.read_csv(file, sep = '\t', header = 0)
            self.data = tmp[[self.config['GENE_COUNT']['geneID'],
                             self.config['GENE_COUNT']['count']]]
            del (tmp)
            self.data.rename(columns = {self.config['GENE_COUNT']['geneID'] : 'geneID',
                                        self.config['GENE_COUNT']['count'] : 'count'}, inplace = True)


            self.logger.info (f"Gene counts for {len (self.data)} genes loaded.")
        except FileNotFoundError:
            self.logger.critical (f"File {file} not found.")
            sys.exit (1)

        try:
            self.reference = pd.read_csv (reference_panel, sep = '\t')
            self.logger.info (f'{len(self.reference)} reference genes read in')
            excluded_list = self.config['ARMS']['excluded_arms'].split (',')
            self.reference = self.reference.loc[[a not in excluded_list for a in self.reference['arm'].tolist()]].copy()
        except FileNotFoundError:
            self.logger.critical (f"File {reference_panel} not found.")
            sys.exit(1)
        
        
    def calculate_ratios (self, data_type, apply_log_to_coverage = False):
        """Calculate normalized coverage values in ``self.data``.

        WGS data is normalized by its median coverage.  Other supported data
        types are merged with the reference panel and normalized with TMM.
        This method adds or updates ``l2r`` and related intermediate columns.
        """
        if data_type in ['wgs']:
            self.median_cv = self.data.loc[self.data['cv'] > 0, 'cv'].median()
            self.data['l2r'] = transform_ratio(self.data['cv']/self.median_cv,
                                               apply_log_to_coverage = apply_log_to_coverage)
            
        elif data_type in ['wes', 'rna', 'atac','chip']:
            self.logger.info (f'Calculating ratios for {data_type} data.')
            self.data = self.data.merge (self.reference, how = 'inner', on = 'geneID')
            
            self.fTMM = TMM (self.data['count'], self.reference['refcpm'])
            self.logger.info (f"Median coverage (1): {self.data['count'].median()}")
            self.logger.info (f'TMM factor: {self.fTMM}')
            self.data['countTMM'] = self.data['count']*2**self.fTMM
            self.data['lm'] = np.log2 (10**6*(self.data['countTMM']+0.5)/(self.data['countTMM'].sum() + 1))
            
            self.data['lmp'] = np.log2 (10**6*(self.data['refcpm'].astype(float).values+0.5)/(self.data['refcpm'].astype(float).values.sum() + 1))
            self.data = self.data.loc[(self.data['lm'] > float(self.config['GENE_COUNT']['cpm_thr']))&\
                                      (self.data['lmp'] > float(self.config['GENE_COUNT']['cpm_thr']))].copy()
            self.logger.info (f'{len(self.data)} genes after filtering.')
            self.data['l2r'] = transform_ratio (self.data['lm'].values/self.data['lmp'].values,
                                                apply_log_to_coverage = apply_log_to_coverage)
            self.data = self.data.loc[self.data['lm'] > float(self.config['GENE_COUNT']['cpm_thr'])].copy()
            self.median_cv = self.data['count'].median()
            self.data.rename (columns = {'count' : 'cv'}, inplace = True)


    def get_data (self) -> pd.DataFrame:
        """Return the processed gene-count data sorted by arm and position."""
        tmp = self.data#.loc[(self.data['arm'] == arm)]
        return tmp.sort_values (by = ['arm', 'Pos']) #loc[(tmp['Pos'] >= start) & (tmp['Pos'] <= end)].


    def find_segments (self, skip_segmentation = False):
        """Create coverage segments from the processed gene-count data.

        Markers with coverage at or below one are excluded from segmentation.
        The resulting :class:`Segments.Segment` object is stored in
        ``self.segments``; ``skip_segmentation`` uses chromosome arms as
        segments instead.
        """
        self.data['Chr'] = [a[:-1] for a in self.data['arm'].tolist()]
        
        self.logger.info('Starting cv segmentation.')
        if any (self.data['cv'] <= 1):
            self.filtered_data = self.data.loc[self.data['cv'] > 1]
            self.logger.info(f'Some markers have cv <= 1. Filtered to {len(self.filtered_data)} records.')
        else:
            self.filtered_data = self.data
            self.logger.info (f'There are {len(self.filtered_data)} markers passed to segmentation.')
        
        
        self.segments = Segments.Segment(data = self.filtered_data.sort_values (by = ['arm', 'Pos']),
                                         signal_column = 'l2r', 
                                         logger = self.logger, msnfun = self.msnfun,
                                         merge_thr = float(self.config['SEGMENTATION']['gc_thr']),
                                         abs_thr = float(self.config['SEGMENTATION']['absolute_counts_threshold'])/np.sqrt(self.median_cv),
                                         config = self.config,
                                         f0 = float(self.config['SEGMENTATION']['gc_p0']),
                                         length = int(self.config['SEGMENTATION']['gc_length']),
                                         ms = float(self.config['SEGMENTATION']['gc_ms']),
                                         outlier_analysis = True,
                                         skip_segmentation = skip_segmentation)

    
    def get_segments (self):
        """Return the segment table produced by :meth:`find_segments`."""
        return self.segments.get_segments()


    def get_params (self, arm, start, end):
        """Return the robust median and spread for one genomic interval.

        Returns ``(numpy.nan, numpy.nan)`` when no data falls within the
        half-open interval ``[start, end)``.
        """
        tmp = self.data.loc[(self.data['arm'] == arm)]
        tmp = tmp.loc[(tmp['Pos'] >= start) & (tmp['Pos'] < end)]
        if len(tmp):
            m, s, n = msnfun(tmp)
            return (m, s)
        else:
            return (np.nan, np.nan)


    def generate_report (self, sample_name, report_name):
        """Write the requested gene-count or coverage report to disk."""
        if report_name == 'gene_counts':
            self.data.to_csv (sample_name + '_' + report_name + '.tsv', sep = '\t', index = False)
        elif report_name == 'reference':
            fg, ax = plt.subplots (1,1, figsize = (16,4))
            self.data.boxplot (column = 'l2r', by = 'arm', ax = ax, rot = 60)
            fg.savefig (sample_name + '_' + report_name + '_reference.jpg')
            plt.close(fg)    
        elif report_name == 'gc_segs':
            segs = self.segments.get_segments()
            armssegs = []
            for arm in segs.keys():
                armseg = segs[arm]
                armseg['arm'] = arm
                armssegs.append (armseg)
            pd.concat (armssegs).to_csv (sample_name + '_' + report_name + '.tsv', sep = '\t', index = False)
            #self.segments.get_segments().to_csv (sample_name + '_' + report_name + '.tsv', sep = '\t', index = False)       


def msnfun (x):
    """Return median, robust spread, and count for a frame with ``l2r``."""
    gc = x['l2r'].values
    s1, m, s2 = np.quantile (gc, q = [0.25, 0.5, 0.75])
    s = np.max(((s2-s1)/1.349, 0.01))
    if np.isnan(m):
        raise ValueError ('Median is NaN')
    return m, s, len(gc)
            

def transform_ratio (ratio, apply_log_to_coverage = False):
    """Return a ratio unchanged or transformed to base-2 logarithms."""
    if not apply_log_to_coverage:
        return ratio
    else:
        return np.log2(ratio)


def TMM (ref, sampl, Acut = TMM_Acut, Mcut = TMM_Mcut):
    """Estimate the trimmed-mean-of-M-values normalization factor.

    Zero-count pairs are removed, then the configured fractions of extreme
    M and A values are trimmed before the weighted factor is calculated.
    """
    Nr = ref.sum()
    Nk = sampl.sum()
    
    init_filt = (ref > 0)&(sampl > 0)
    
    M = np.log2(sampl[init_filt]/Nk) / np.log2(ref[init_filt]/Nr)
    A = np.log2(np.sqrt(ref[init_filt]*sampl[init_filt]/(Nr*Nk)))
    
    Md, Mu = np.quantile (M, [Mcut, 1-Mcut])
    Ad, Au = np.quantile (A, [Acut, 1-Acut])
    
    filt = (M >= Md)&(M <= Mu)&(A >= Ad)&(A <= Au)
    
    sf = sampl[init_filt][filt]
    rf = ref[init_filt][filt]
    
    Nr = rf.sum()
    Nk = sf.sum()
    Mf = np.log2(sf/Nk) / np.log2(rf/Nr)
    Af = np.log2(np.sqrt(rf*sf/(Nr*Nk)))
    wf = (Nk - sf)/(Nk*sf) + (Nr-rf)/(Nr*rf)
    
    f = (np.sum(wf*Mf)/np.sum(wf))
    
    return f

