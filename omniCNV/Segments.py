import numpy as np
import pandas as pd
import scipy.stats as sts
import scipy.optimize as opt

from typing import List, Tuple, Iterable, Sequence
from collections import Counter

from Fryzlewicz.base_search import generate_haar_basis
from Fryzlewicz.decomposition import decompose, rle_to_array

class Segment:
    """
    Performs segmentation based on Fryzlewicz transformation of genomic 
    signals (ai or coverage), followed by greedy segment merging.

    Parameters
    ----------
    data : pd.DataFrame  Must contain columns 'Chr', 'arm', 'Pos', and signal_column.
    signal_column : str  Column name to segment ('ai' or 'l2r').
    msnfun : callable  Function (data) -> (mean, std, n) for a segment.
    merge_thr : float  Z-score threshold below which adjacent segments are merged.
    abs_thr : float  Absolute difference threshold; segments closer than this are merged.
    outlier_analysis : bool  If True, run a secondary outlier-splitting pass (intended for coverage).
    include_HO : bool  If False and signal_column is 'ai', exclude rows with vaf == 1 from segmentation.
    """
    def __init__(self, data, logger, msnfun, signal_column, merge_thr,
                 abs_thr, config, f0 = 0.8, length = 21, ms = 3,
                 outlier_analysis = False, include_HO = False, skip_segmentation = False):
        
        self.logger = logger.getChild (f'{self.__class__.__name__}')
        self.data = data
        self.f0 = f0
        self.length = length
        self.ms = ms
        self.config = config
        self.signal_column = signal_column
        self.include_HO = include_HO
        self.msn = msnfun
        self.logger.debug (f"""Initializing {self.__class__.__name__} with signal column {signal_column},
                            f0: {self.f0}, length: {self.length}, ms: {self.ms}.""")
        self.thr = merge_thr
        self.abs_thr = abs_thr
        
        if skip_segmentation:
            self.logger.info (f'Skipping segmentation, using chromosome arms as segments.')
            self.make_arm_segments()
            
        else:
            self.logger.info (f'Finding segments') 
            self.find_segments()
            self.logger.info (f'Merging segments')
            self.merge_segments()
            del (self.raw_segments)
        
        if outlier_analysis:
            self.logger.info (f'Analyzing outliers within each segment')
            self.analyze_outliers()
        del (self.data)


    def make_arm_segments (self):
        """Create one segment for each chromosome arm.

        This is the fallback used when segmentation is skipped.  Results are
        stored in ``self.segments`` as a mapping from arm name to DataFrame.
        """
        self.segments = {}
        for arm, armdata in self.data.groupby(by = 'arm'):
                result = ((np.nan, armdata.shape[0], armdata['Pos'].min(), armdata['Pos'].max(), *self.msn(armdata)),)
                self.segments[arm] = pd.DataFrame.from_records (result,
                                                                columns = ['signal', 'len', 'start', 'end', 'm', 's', 'n'])


    def find_segments(self):
        """
        Build initial raw segments per chromosome arm and store them in self.raw_segments.

        Workflow
        --------
        1. Optionally filter out rows with vaf >= 1 if a vaf column exists.
        2. Process data chromosome by chromosome.
        3. If any arm has too few markers, segment the whole chromosome when possible
           and clip resulting segments to arm boundaries; otherwise create one fallback
           segment per arm.
        4. If all arms meet the minimum marker threshold, segment each arm independently.

        Output format
        -------------
        self.raw_segments is a dict:
            arm -> list of [signal, len, start, end]
        """
        segments = {}
        min_markers = float(self.config['SEGMENTATION']['min_markers_for_seg'])

        required_columns = {'Chr', 'arm', 'Pos', self.signal_column}
        missing_columns = required_columns.difference(self.data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns for segmentation: {sorted(missing_columns)}")

        if self.signal_column == 'ai':
            if  self.include_HO:
                fractions = self.data.groupby(by = 'arm').agg({'vaf': lambda x: (x == 1).sum()/len(x)}).reset_index()
                minHOfraction = fractions['vaf'].min()
                self.logger.info (f'Min fHO = {minHOfraction:.3f} on arm {fractions.loc[fractions["vaf"].idxmin(), "arm"]}.')
                tmp = self.data
            else:
                tmp = self.data.loc[self.data['vaf'] < 1]
                self.logger.info('Excluding homozygous markers (vaf >= 1) from ai segmentation.')
                minHOfraction = 0
        else:
            minHOfraction = 0
            tmp = self.data
        
        self.logger.info (f'Finding segments with min_markers: {min_markers}, minHOfraction: {minHOfraction:.3f}')
        for chrom, chromdata in tmp.groupby(by = 'Chr'):
            chromdata = chromdata.sort_values(by = 'Pos')
            if any(chromdata.groupby(by = 'arm').size() < min_markers):
                if chromdata.shape[0] >= min_markers:
                    self.logger.info (f'Chromosome {chrom} segmented as whole')
                    chrom_segs = []
                    values = chromdata[self.signal_column].values
                    positions = chromdata['Pos'].values
                    basis = generate_haar_basis (values, self.f0, self.length, fHO = minHOfraction)
                    self.logger.debug(f'Basis length: {len(basis)}')
                    results = decompose (values, basis, ms = self.ms)
                    start = 0
                    for rle in results['rle']:
                        end = start + rle[1]
                        chrom_segs.append([rle[0], rle[1], positions[start], positions[end-1]])
                        start = end

                    # already two arms present
                    arms = chromdata.groupby(by = 'arm').agg({'Pos': ['min', 'max']})['Pos'].reset_index()
                    for _, arm in arms.iterrows():
                        arm_name = arm['arm']
                        arm_min = arm['min']
                        arm_max = arm['max']
                        segments[arm_name] = []
                        for seg in chrom_segs:
                            tmpseg = seg.copy()
                            if seg[2] < arm_min:
                                tmpseg[2] = arm_min
                            if seg[3] > arm_max:
                                tmpseg[3] = arm_max
                            if (tmpseg[2] <= tmpseg[3]) and (tmpseg[2] >= arm_min) and (tmpseg[3] <= arm_max):
                                segments[arm_name].append(tmpseg)
                        self.logger.info (f'{len(segments[arm_name])} segments on {arm_name} (1)')
                else:
                    self.logger.info (f'Chromosome {chrom} skipped for segmentation, too few markers: {chromdata.shape[0]}.')
                    for arm, armdata in chromdata.groupby(by = 'arm'):
                        # adding single fallback segment for this arm
                        segments[arm] = [[0, armdata.shape[0], armdata['Pos'].min(), armdata['Pos'].max()]]
            else:
                for arm, armdata in chromdata.groupby(by = 'arm'):
                    segments[arm] = []
                    armdata = armdata.sort_values(by = 'Pos')
                    values = armdata[self.signal_column].values
                    positions = armdata['Pos'].values
                    basis = generate_haar_basis (values, self.f0, self.length, fHO = minHOfraction)

                    self.logger.debug(f'Basis length: {len(basis)}')
                    results = decompose (values, basis, ms = self.ms)

                    start = 0
                    for rle in results['rle']:
                        end = start + rle[1]
                        segments[arm].append([rle[0], rle[1], positions[start], positions[end-1]])
                        start = end

                    self.logger.info (f'{len(segments[arm])} segments on {arm}')

        self.raw_segments = segments


    def merge_segments(self):
        """Greedily merge adjacent raw segments with similar signal values.

        Merging uses both the configured z-score threshold and ``abs_thr``.
        Final segment tables are stored in ``self.segments`` by chromosome arm.
        """
        self.segments = {}
        
        for arm in self.raw_segments.keys():
            rln = self.raw_segments[arm].copy()
            self.logger.debug (f'Starting segments on {arm} with {len(rln)} segments')
            #self.logger.debug (f'Starting segments on {arm}: {rln}')
            armdata = self.data.loc[self.data.arm == arm].sort_values(by = 'Pos')
            ziff, m, s, n = calculate_differences (rln,
                                                   armdata,
                                                   self.msn, thr = self.abs_thr)
            if len(rln) > 1:
                self.logger.debug (f'Initial segments on {arm} differ by {ziff}, n: {n}')
                armdata = self.data.loc[self.data.arm == arm].sort_values(by = 'Pos')
                #TODO: make threshold dynamic based on data
                while (min(ziff) < self.thr) | any(np.abs(m[:-1]-m[1:]) < self.abs_thr) | any (np.isnan(ziff)):
                    if any (np.isnan(ziff)):
                        indmin = np.where (np.isnan(ziff))[0][0]
                    elif (min(ziff) < self.thr):
                        indmin = np.where (ziff == np.min(ziff))[0][0]
                    else:
                        d = np.abs(m[:-1]-m[1:])
                        indmin = np.where (d == np.min(d))[0][0]
                    
                    merged = [np.nan, rln[indmin][1]+rln[indmin+1][1],
                              rln[indmin][2], rln[indmin+1][3]] 
                    rln [indmin] = merged
                    del (rln[indmin+1])
                    #ziff = np.delete (ziff, indmin)
                    m = np.delete (m, indmin+1)
                    s = np.delete (s, indmin+1)
                    n = np.delete (n, indmin+1)
                    _, ml, sl, nl = calculate_differences ([rln[indmin]],
                                                            armdata,
                                                            self.msn,
                                                            thr = self.abs_thr)
                    m[indmin] = ml[0]
                    s[indmin] = sl[0]
                    n[indmin] = nl[0]
                    ziff = np.abs(m[:-1]-m[1:])/np.sqrt((s[:-1]**2)/n[:-1] + (s[1:]**2)/n[1:] + self.abs_thr**2)
                    
                    if len (rln) == 1:
                        break
                
                #TODO: check for differences in m, cased by 1 read
                self.logger.debug (f'Final segments on {arm} differ by {ziff}, n: {n}')
                self.segments[arm] = pd.DataFrame.from_records ([[*ri, mi, si, ni] for ri, mi, si, ni in zip (rln,m,s,n)],
                                                                columns = ['signal', 'len', 'start', 'end', 'm', 's', 'n'])
            elif len(rln) == 1:
                self.logger.debug (f'Segment on {arm} parameters: n = {n}')
                self.segments[arm] = pd.DataFrame.from_records ([[*rln[0], m[0], s[0], n[0]]],
                                                                columns = ['signal', 'len', 'start', 'end', 'm', 's', 'n'])
                
            self.logger.info (f'{len(self.segments[arm])} segments found on {arm}')
            

    def analyze_outliers(self):
        """Split coverage segments around paired, strong local outliers.

        The replacement segment tables retain the standard ``signal``, ``len``,
        ``start``, ``end``, ``m``, ``s``, and ``n`` columns.
        """
        #self.outliers = {}
        new_segs = {}
        for arm in self.segments.keys():
            self.logger.debug (f'Segmenting through outliers on {arm}')
            segs = self.segments[arm]
            
            records = []
            for _, seg in segs.iterrows():
                out_segs = []
                segdata = self.data.loc[(self.data.arm == arm)&\
                                        (self.data['Pos'] >= seg['start']) &\
                                        (self.data['Pos'] <= seg['end'])].sort_values(by = 'Pos')
                d = (segdata['cv'].values[1:] - segdata['cv'].values[:-1])/np.sqrt(segdata['cv'].mean()*2)
                #TODO: move '3' to config
                indpl = np.where (d > 3)[0]
                indmn = np.where (d < -3)[0]
                self.logger.debug (f'Found {len(indpl)} positive and {len(indmn)} negative outliers in segment {arm} {seg["start"]}-{seg["end"]}')
                if len(indpl) & len(indmn):
                    outlier_indexes = find_pairs_directional_unique (segdata['Pos'].values,
                                                                     indpl,
                                                                     indmn,
                                                                     min_index_gap = 3,
                                                                     max_pos_gap = float(self.config['KARYOTYPING']['MIN_SIZE'])*1e6)
                    self.logger.debug (f'Found {len(outlier_indexes)} outlier correct pairs: {outlier_indexes}.')
                    for (i,j) in outlier_indexes:
                        k,l = np.sort ((i,j))
                        zs = (segdata.iloc[k+1:l+1]['l2r'].values - seg['m'])/np.sqrt(seg['s'])
                        #TODO: Move '3' to config
                        if np.abs (np.median(zs)) > 3:
                            out_segs.append (k)
                            out_segs.append (l)
                
                out_segs = np.sort (np.unique(out_segs))
                self.logger.debug (f'There are {len(out_segs)/2} strong outlier segments')        
                if len (out_segs):
                    starts = []
                    ends = []
                    
                    #if out_segs[0] > 0:
                    starts.append (segdata['Pos'].values[0])

                    for i in out_segs:
                        ends.append (segdata['Pos'].values[i+1])
                        if i+2 < segdata.shape[0]:
                            starts.append (segdata['Pos'].values[i+2])
                    
                    if ends[-1] < segdata['Pos'].values[-1]:
                        ends.append (segdata['Pos'].values[-1])
                    
                    for st,en in zip(starts, ends):
                        start, end = np.sort ((st, en))
                        tmpdata = segdata.loc[(segdata['Pos'] >= start) & (segdata['Pos'] <= end)]
                        m, s, n = self.msn (tmpdata)
                        records.append ((np.nan, end-start, start, end, m, s, n))
                        self.logger.debug (f'Adding segment {arm} {start}-{end} with m: {m}, s: {s}, n: {n}')
                        
                else:   
                    records.append ((*seg[['signal', 'len', 'start', 'end', 'm', 's', 'n']].values,))
                    self.logger.debug (f'Coping segment {arm} {seg["start"]}-{seg["end"]} with m: {seg["m"]}, s: {seg["s"]}, n: {seg["n"]}')
            if len (records):
                self.logger.info (f'Adding {len(records)} outlier segments to {arm}')
                self.logger.debug (f'New segments on {arm}: {records}')
                new_segs[arm] = pd.DataFrame.from_records (records,
                                                    columns = ['signal', 'len', 'start', 'end', 'm', 's', 'n'])
            else:
                self.logger.debug (f'No outlier segments found for {arm}')
                 
        self.segments = new_segs
                        
    def get_segments(self):
        """Return the final segment mapping keyed by chromosome arm."""
        return self.segments

    #left for back-compatibility
    def get_raw_segments(self):
        """Return the segment mapping for backward compatibility."""
        return self.segments


def find_pairs_directional_unique(positions: Sequence[float],
                                  idx_a: Iterable[int],
                                  idx_b: Iterable[int],
                                  min_index_gap: int = 3,
                                  max_pos_gap: float = 10000) -> List[Tuple[int, int]]:
    """
    Find unambiguous one-to-one pairs (i, j) with:
        i ∈ idx_a, j ∈ idx_b
        abs(i - j) >= min_index_gap
        positions[j] > positions[i]
        positions[j] - positions[i] <= max_pos_gap

    Additional uniqueness rule:
        - If an i has multiple valid j's, reject all pairs involving that i.
        - If a j has multiple valid i's, reject all pairs involving that j.

    Returns only pairs (i, j) where i has exactly one valid j and j has exactly one valid i.
    """
    idx_a = list(idx_a)
    idx_b = list(idx_b)

    # 1) Build candidate pairs that meet all directional + gap constraints.
    candidates: List[Tuple[int, int]] = []
    for i in idx_a:
        pi = positions[i]
        for j in idx_b:
            if abs(i - j) < min_index_gap:
                continue
            pj = positions[j]
            if pj <= pi:
                continue  # directional: B must be strictly larger than A
            if pj - pi <= max_pos_gap:
                candidates.append((i, j))
    
    if not candidates:
        return []

    # 2) Count how many partners each index appears with.
    count_i = Counter(i for i, _ in candidates)
    count_j = Counter(j for _, j in candidates)

    # 3) Keep only pairs where both sides are uniquely matched.
    result = [(i, j) for (i, j) in candidates if count_i[i] == 1 and count_j[j] == 1]
    return result


def calculate_differences (rle, data, msnfun, thr = 0): 
    """Calculate signal differences between adjacent segments and per-segment statistics.

    Returns ``(z_scores, medians, spreads, counts)`` for the run-length
    encoded intervals in ``rle``.
    """
    ms = []
    ss = []
    ns = []

    for _,_,start,end in rle:
        tmp = data.loc[(data['Pos'] >= start) & (data['Pos'] <= end)]
        
        ns.append(len(tmp))
        m, s, n = msnfun(tmp)
        
        ms.append (m) 
        ss.append (s) 

    m = np.array (ms, dtype = float)
    s = np.array (ss, dtype = float)
    n = np.array (ns, dtype = float)
    
    return np.abs(m[:-1]-m[1:])/np.sqrt((s[:-1]**2)/n[:-1] + (s[1:]**2)/n[1:] + thr**2), m, s, n


