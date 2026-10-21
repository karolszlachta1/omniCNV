import sys 
import numpy as np
import pandas as pd
import scipy.stats as sts
import scipy.optimize as opt

from typing import List, Tuple, Iterable, Sequence
from collections import Counter


from Fryzlewicz.base_search import generate_haar_basis
from Fryzlewicz.decomposition import decompose, rle_to_array

class chrY:
    """Class for handling chromosome Y data, including filtering, arm assignment, and segmentation."""
    def __init__(self, data, logger, config, skip_segmentation = False):
        """Filter chromosome-Y markers, assign arms, and build segments.

        When ``skip_segmentation`` is true, one median-coverage segment is
        created per arm; otherwise each arm is segmented with the Haar-wavelet
        workflow.
        """
        self.logger = logger.getChild(f'{self.__class__.__name__}')
        self.config = config
        self.skip_segmentation = skip_segmentation
        
        self.logger.info ('Initializing chrY')
        self.filter_data(data)
        self.assign_arm ()
        
        
        if skip_segmentation:
            self.logger.info ('Skipping segmentation for chrY')
            segments = []
            for arm, grdata in self.data.groupby('arm'):
                q2, q1 = np.quantile(grdata['cv'], [0.75, 0.25])
                segments.append({'arm': arm,
                                 'start': grdata['Pos'].min(),
                                 'end': grdata['Pos'].max(),
                                 'cv': grdata['cv'].median(),
                                 'dcv': (q2-q1)/1.349,
                                 'ai' : 0.5,
                                 'dai': np.nan})
            self.segments = pd.DataFrame(segments)
            
        else:
            try:
                segp = self.find_segments('chrYp')
            except:
                segp = pd.DataFrame(columns = ['arm', 'start', 'end', 'cv', 'dcv', 'ai', 'dai'])
                segp = segp.astype({'arm': 'string',
                                    'start': 'int64',
                                    'end': 'int64',
                                    'cv': 'float64',
                                    'dcv': 'float64',
                                    'ai': 'float64',
                                    'dai': 'float64'})
            try:
                segq = self.find_segments('chrYq')
            
            except:
                segq = pd.DataFrame(columns = ['arm', 'start', 'end', 'cv', 'dcv', 'ai', 'dai'])
                segq = segq.astype({'arm': 'string',
                                    'start': 'int64',
                                    'end': 'int64',
                                    'cv': 'float64',
                                    'dcv': 'float64',
                                    'ai': 'float64',
                                    'dai': 'float64'})
            
            self.segments = pd.concat([segp, segq], ignore_index = True)
        
        self.logger.info (f'{len(self.segments)} segments found for chrY.')


    def assign_arm (self) -> None:
        """ Assigns chromosomal arms to each marker based on a provided arms file. 
        The arms file should have columns 'Chr' and 'Boundry', where 'Boundry' is the 
        position separating p and q arms. Markers are labeled as 'chrXp' or 'chrXq' accordingly.
        Arms listed in the config's excluded_arms are filtered out."""
        try:
            arms = pd.read_csv (self.config['FILES']['arms'], sep = '\t')
        except FileNotFoundError:
            self.logger.critical (f"Chrom arms file {self.config['FILES']['arms']} not found.")
            sys.exit(1)
        
        self.data['arm'] = ''
        chrom = 'chrY'
        boundry = arms.loc[arms.Chr == chrom, 'Boundry'].values[0]
        self.data['arm'] = [chrom+'p' if p < boundry else chrom+'q' for p in self.data['Pos']]
                
        del(arms)


    def filter_data(self, data):
        """Remove PAR and non-HO markers and store derived marker fields.

        The resulting ``self.data`` contains ``cv``, ``vaf``, ``ai``.
        """
        # Filter out PAR regions
        data['cv'] = (data['ref_cts'] + data['alt_cts'])
        data['vaf'] = data['alt_cts'] / data['cv']
        data['ai'] = np.abs(data['vaf'] - 0.5)
        
        PAR = eval(self.config['chrY']['PAR'])
        filtered_data = data.loc[(data['Pos'] > PAR[0][1]) & (data['Pos'] < PAR[1][0]) &\
                                 (data['vaf'] >= float (self.config['chrY']['vaf_min']))]
        filtered_data['arm'] = 'chrY '
        self.data = filtered_data.copy()

    def find_segments(self, arm):
        """Segment one chromosome-Y arm by coverage and return a DataFrame.

        Returned columns are ``arm``, ``start``, ``end``, ``cv``, ``dcv``,
        ``ai``, and ``dai``.  Chromosome-Y segments use ``ai=0.5`` and a small
        fixed ``dai`` because allele imbalance is not estimated here.
        """
        # Segment the data using Haar wavelet transform
        data = self.data.loc[self.data['arm'] == arm]
        self.logger.info (f'Segmenting {arm} data using Haar wavelet transform.')
        positions = data['Pos'].values
        values = data['cv'].values
        basis = generate_haar_basis(values,
                                    p0 = float(self.config['SEGMENTATION']['gc_p0']),
                                    length = int(self.config['SEGMENTATION']['gc_length']))
        results = decompose (values, basis, ms = float(self.config['SEGMENTATION']['gc_ms']))
        
        start = 0
        raw_segments = []
        
        starts = []
        ends = []
        mcv = []
        scv = []

        for rle in results['rle']:
            end = start + rle[1]
            q2, q1 = np.quantile(values[start:end], [0.75, 0.25])
            starts.append(start)
            ends.append(end)
            mcv.append(np.median(values[start:end]))
            scv.append((q2-q1)/1.349)
            start = end

        m = np.array(mcv)
        s = np.array(scv)
        n = np.array([end-start for start, end in zip(starts, ends)])
        
        if len(m) > 1:
            abs_thr = float(self.config['SEGMENTATION']['absolute_counts_threshold'])
            merge_thr = float(self.config['SEGMENTATION']['gc_thr'])
            zdiff = np.abs(m[:-1]-m[1:])/np.sqrt((s[:-1]**2)/n[:-1] + (s[1:]**2)/n[1:] + abs_thr**2)

            while (np.min(zdiff) < merge_thr) or (any(np.abs(m[:-1]-m[1:]) < abs_thr)) or (any (np.isnan(zdiff))):
            
                if any (np.isnan(zdiff)):
                    indmin = np.where (np.isnan(zdiff))[0][0]
                elif (np.min(zdiff) < merge_thr):
                    indmin = np.where (zdiff == np.min(zdiff))[0][0]
                else:
                    d = np.abs(m[:-1]-m[1:])
                    indmin = np.where (d == np.min(d))[0][0]
                m = np.delete (m, indmin+1)
                s = np.delete (s, indmin+1)
                
                starts.pop(indmin+1)
                ends[indmin] = ends[indmin+1]
                ends.pop(indmin+1)

                n = np.array([end-start for start, end in zip(starts, ends)])
                zdiff = np.abs(m[:-1]-m[1:])/np.sqrt((s[:-1]**2)/n[:-1] + (s[1:]**2)/n[1:] + abs_thr**2)
                        
                if len (starts) == 1:
                    break

        segments = []
        for start, end, mcv, scv in zip(starts, ends, m, s):
            segments.append((arm,
                             positions[start],
                             positions[end-1],
                             mcv,
                             scv,
                             0.5, #ai
                             10**-6)) #dai

        return pd.DataFrame.from_records(segments,
                                         columns = ['arm', 'start', 'end', 'cv', 'dcv', 'ai', 'dai'])
        

    def get_segments(self):
        """Return the combined chromosome-Y segment DataFrame."""
        return self.segments


    #left for back-compatibility
    def get_raw_segments(self):
        """Return chromosome-Y segments for backward compatibility."""
        return self.segments


    def generate_report(self, sample_name):
        """Write chromosome-Y segments to ``<sample>_chrY_report.tsv``."""
        report_file = f'{sample_name}_chrY_report.tsv'        
        self.segments.to_csv(report_file, sep = '\t', index = False)
        self.logger.info (f'chrY report saved to {report_file}.')
