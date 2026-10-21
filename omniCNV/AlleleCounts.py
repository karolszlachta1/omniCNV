import pandas as pd
import gzip as gz
import numpy as np
import sys as sys
import numpy as np
import scipy.optimize as opt
import scipy.stats as sts
import io
import polars as pl
        
from omniCNV import Segments as Segments
from omniCNV import chrY


COLUMNS = ['Chr', 'Pos', 'ref_cts', 'alt_cts', 'Type', 'ref', 'alt']
NUCLEOTIDES = set(['A', 'C', 'G', 'T'])

class AlleleCounts:
    """
    Reads allele counts from a TSV or VCF file, filters and segments them, and provides access to the processed data.
    """
    def __init__(self, data_file, input_type, config, logger, adaptive_cvthr = False,
                 include_HO = False, skip_segmentation = False,
                 analyze_chrY = False, skip_filter_good_SNP = False):
        """Load allele counts, calculate allele imbalance, and perform segmentation.

        ``input_type`` must be ``"allele_counts"`` or ``"vcf"``.  The
        processed marker table is stored in ``self.data`` with ``cv``, ``vaf``,
        ``ai``, and ``arm`` columns; segmentation results are available through
        :meth:`get_segments`.
        """
        
        self.config = config
        self.logger = logger.getChild (f'{self.__class__.__name__}')
        self.adaptive_cvthr = adaptive_cvthr
        self.include_HO = include_HO
        self.analyze_chrY = analyze_chrY
        self.skip_filter_good_SNP = skip_filter_good_SNP
        # Load the allele counts data
        if input_type == 'allele_counts':
            self.logger.info (f'Loading allele counts from {data_file}')
            self.fetch_ac (data_file)
        elif input_type == 'vcf':
            self.logger.info (f'Loading VCF from {data_file}')
            self.fetch_vcf (data_file)
        else:
            raise ValueError (f'Unknown input type: {input_type}')
        
        self.logger.info (f'Loaded {len(self.all_data)} records.')
        
        if self.analyze_chrY:
            self.logger.info ('Analyzing chromosome Y.')
            self.chrY = chrY.chrY(self.all_data.loc[self.all_data['Chr'] == 'chrY'].copy(),
                                  logger = logger,
                                  config = config,
                                  skip_segmentation = skip_segmentation)
        else:
            self.chrY = None

        if self.skip_filter_good_SNP:
            self.logger.info ('Skipping filtering of good SNPs.')
            self.data = self.all_data
        else:
            self.logger.info ('Filtering data based on markers list.')
            self.filter_data ()

        self.assign_arm ()
        
        self.data['cv'] = (self.data['ref_cts'] + self.data['alt_cts'])
        self.data['vaf'] = self.data['alt_cts'] / self.data['cv']
        self.data['ai'] = np.abs(0.5 - self.data['vaf'])

        self.logger.info (f'Filtered data to {len(self.data)} records.')
        num_markers = len(self.data.loc[self.data['cv'] >= float(self.config['AC_FILTER']['cvthr'])])
        
        if num_markers < 2:
            self.logger.critical (f"""Not enough markers with sufficient coverage after filtering.
                                   Median coverage {self.data['cv'].median()}.""")
            sys.exit(1)
        if num_markers < int(self.config['PERFORMANCE']['min_markers']):
            self.logger.warning (f"""Not enough markers with sufficient coverage after filtering.
                                   Median coverage {self.data['cv'].median()}.""")

        self.median_cv = self.data['cv'].median()
        
        self.find_segments (skip_segmentation = skip_segmentation)


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
        chroms = set(arms['Chr'].unique())
        tmp = self.data.loc[[a in chroms for a in self.data['Chr'].tolist()]].copy()
        del (self.data)
        self.data = tmp
        
        self.data['arm'] = ''
        for chrom, dc in self.data.groupby (by = 'Chr'):
            boundry = arms.loc[arms.Chr == chrom, 'Boundry'].values[0]
            arm = [chrom+'p' if p < boundry else chrom+'q' for p in dc['Pos']]
            self.data.loc[dc.index , 'arm'] = arm
        
        del(arms)

        excluded_list = self.config['ARMS']['excluded_arms'].split (',')
        self.data = self.data.loc[[a not in excluded_list for a in self.data['arm'].tolist()]].copy()


    def filter_data(self) -> None:
        """
        Polars-based streaming semi-join:
        Keeps rows in self.all_data that match any (Chr, Pos) from markers TSV.
        Returns result as pandas DataFrame in self.data.
        """

        try:
            ad_pl = pl.from_pandas(self.all_data)
            ad_pl = ad_pl.with_columns(
                pl.col("Chr").cast(pl.Utf8),
                pl.col("Pos").cast(pl.UInt32),
            )

            sg_scan = pl.scan_csv(
                self.config["FILES"]["markerslist"],
                separator="\t",
                has_header=False,
                new_columns=["Chr", "Pos"],
                dtypes={"Chr": pl.Utf8, "Pos": pl.UInt32},
                ignore_errors=False,
            )

            # Semi-join: keep only rows from all_data that appear in sg on (Chr, Pos)
            # Collect with streaming=True so it doesn't materialize the full markers table in RAM
            result_pl = (
                ad_pl.lazy()
                .join(sg_scan, on=["Chr", "Pos"], how="semi")
                .collect(streaming=True)
            )
            self.data = result_pl.to_pandas(use_pyarrow_extension_array = False)
            del (result_pl)

            
            #result_anti = (
            #    ad_pl.lazy()
            #    .join(sg_scan, on=["Chr", "Pos"], how="anti")
            #    .collect(streaming=True)
            #)
            
            #self.somatic_markers = result_anti.to_pandas(use_pyarrow_extension_array = False)
            # Free some memory
            del (self.all_data)
            del (sg_scan)
            del (ad_pl)
            
            #del (result_anti)

            self.logger.debug(f"Polars semi-join complete. Result rows: {len(self.data)}")
        except FileNotFoundError:
            self.logger.critical(
                f'Markers file {self.config["FILES"]["markerslist"]} not found.'
            )
            sys.exit(1)
        except Exception as e:
            self.logger.exception(f"filter_data (polars) failed: {e}")
            raise


    def fetch_vcf(self, file) -> None:
        """
        Fast & robust VCF parser:
        - Auto-detects gzip via magic number
        - Caches AD index per FORMAT string
        - Sums ALT counts for multi-allelic AD (configurable)
        - Skips malformed/missing AD safely
        - Tracks duplicates globally
        """

        # Config-driven field indices (0-based)
        CHROM_FIELD = int(self.config['VCF_FORMAT']['chrom_field'])
        POSITION_FIELD = int(self.config['VCF_FORMAT']['position_field'])
        REF_FIELD = int(self.config['VCF_FORMAT']['ref_field'])
        ALT_FIELD = int(self.config['VCF_FORMAT']['alt_field'])
        TAG_NAME_FIELD = int(self.config['VCF_FORMAT']['tag_name_field'])  # FORMAT column
        TAG_FIELD = int(self.config['VCF_FORMAT']['tag_field'])            # Sample column

        wrong_records = 0

        # Output columns as separate lists (faster than list of tuples)
        chrs, poses, ref_cts, alt_cts = [], [], [], []

        # Cache: FORMAT string -> AD index
        fmt_cache = {}

        # Duplicate detection
        seen = set()

        try:
            fb = open(file, 'rb')

            with fb:
                # Detect gzip by magic number
                head = fb.peek(2) if hasattr(fb, 'peek') else fb.read(2)
                if not hasattr(fb, 'peek'):
                    fb.seek(0)

                # Build a text stream once
                if head.startswith(b'\x1f\x8b'):
                    tf = io.TextIOWrapper(gz.GzipFile(fileobj=fb), encoding='utf-8', newline='')
                else:
                    tf = io.TextIOWrapper(fb, encoding='utf-8', newline='')

                with tf:
                    for line in tf:
                        if not line or line[0] == '#':
                            continue

                        fields = line.rstrip('\n').split('\t')

                        # Guard for truncated/malformed lines
                        try:
                            chrom = fields[CHROM_FIELD]
                            pos = int(fields[POSITION_FIELD])
                            ref = fields[REF_FIELD]
                            alt = fields[ALT_FIELD]
                            fmt_str = fields[TAG_NAME_FIELD]
                            sample_str = fields[TAG_FIELD]
                        except (IndexError, ValueError):
                            wrong_records += 1
                            continue
                        
                        # Ignore symbolic ALT alleles (for example <NON_REF>) and
                        # validate REF plus the first real ALT nucleotide.
                        alt_alleles = [a for a in alt.split(',') if a and not a.startswith('<')]
                        if (ref not in NUCLEOTIDES) or (len(alt_alleles) == 0) or (alt_alleles[0] not in NUCLEOTIDES):
                            wrong_records += 1
                            continue
                        # Find AD position with cache
                        ADpos = fmt_cache.get(fmt_str)
                        if ADpos is None:
                            fmt_parts = fmt_str.split(':')
                            try:
                                ADpos = fmt_parts.index('AD')
                            except ValueError:
                                # No AD in this FORMAT
                                wrong_records += 1
                                continue
                            fmt_cache[fmt_str] = ADpos

                        # Pull the AD subfield out of the sample field
                        try:
                            ad_field = sample_str.split(':')[ADpos]
                        except IndexError:
                            wrong_records += 1
                            continue

                        if ad_field in ('.', ''):
                            wrong_records += 1
                            continue

                        parts = ad_field.split(',')
                        # POLICY:
                        # - require at least REF and first ALT counts
                        # - allow additional ALT positions only if all are zero
                        #   (for example "10,5,0,0" is accepted)
                        if len(parts) < 2:
                            wrong_records += 1
                            continue
                        if len(parts) > 2:
                            try:
                                extra_alt_nonzero = any(int(p) != 0 for p in parts[2:])
                            except ValueError:
                                wrong_records += 1
                                continue
                            if extra_alt_nonzero:
                                wrong_records += 1
                                continue

                        # Robust int conversion with '.'-handling
                        try:
                            ref = int(parts[0])
                            alt = int(parts[1])
                            #alt = 0
                            #for p in parts[1:]:
                            #    if p not in ('.', ''):
                            #        alt += int(p)
                        except ValueError:
                            wrong_records += 1
                            continue

                        # Duplicate detection (global)
                        key = (chrom, pos)
                        if key in seen:
                            self.logger.warning(f'Duplicated record:\n{chrom}\t{pos}')
                        else:
                            seen.add(key)

                        chrs.append(chrom)
                        poses.append(pos)
                        ref_cts.append(ref)
                        alt_cts.append(alt)
        except Exception as e:
            self.logger.critical(f'Error reading file {file}: {e}')
            import sys
            sys.exit(1)

        if wrong_records:
            self.logger.info(f'There were {wrong_records} records without usable AD')

        # Build DataFrame with precise dtypes
        all_data = pd.DataFrame({
            'Chr': chrs,
            'Pos': pd.Series(poses, dtype='int64'),
            'ref_cts': pd.Series(ref_cts, dtype='int32'),
            'alt_cts': pd.Series(alt_cts, dtype='int32'),
        })


        all_data['Chr'] = np.where(all_data['Chr'].str.contains("chr") == False, "chr"+all_data['Chr'], all_data['Chr'])

        self.all_data = all_data.loc[all_data['alt_cts'] > 0].copy()
        del(all_data)


    def fetch_ac (self, file) -> None:
        """Read a tab-separated allele-count file and retain SNP records.

        Input column names are taken from the ``ALLELIC_COUNT_FORMAT``
        configuration section and normalized to the module-level ``COLUMNS``
        schema in ``self.all_data``.
        """
        try:
            acColumns = [self.config['ALLELIC_COUNT_FORMAT'][c] for c in COLUMNS]
            dataAll = pd.read_csv (file, sep = '\t',  usecols = acColumns)[acColumns]
            dataAll.columns = COLUMNS
            data = dataAll.loc[dataAll.Type == 'SNP'].copy()
            del(dataAll)
            self.all_data = data
        except FileNotFoundError:
            self.logger.critical (f'File {file} not found.')
            sys.exit(1)


    def generate_report (self, sample_name, report_name):
        """Write the requested allele-count, segment, or auxiliary report."""
        if report_name == 'allele_counts':
            self.data.to_csv (sample_name + '_' + report_name + '.tsv', sep = '\t', index = False)
        elif report_name == 'chrY':
            self.chrY.data['l2r'] = self.chrY.data['cv']/self.median_cv
            self.chrY.data[self.data.columns.tolist()].to_csv (sample_name + '_chrY_allele_counts.tsv', sep = '\t', index = False)
        elif report_name == 'ROH':
            runs = self.runs.get_runs()
            armsroh = []
            for arm in runs.keys():
                armroh = runs[arm]
                armroh['arm'] = arm
                armsroh.append (armroh)
            pd.concat (armsroh).to_csv (sample_name + '_' + report_name + '.tsv', sep = '\t', index = False)
        elif report_name == 'ac_segs':
            segs = self.segments.get_raw_segments()
            armssegs = []
            for arm in segs.keys():
                armroh = segs[arm]
                armroh['arm'] = arm
                armssegs.append (armroh)
            pd.concat (armssegs).to_csv (sample_name + '_' + report_name + '.tsv', sep = '\t', index = False)
            if self.chrY is not None:
                self.chrY.generate_report(sample_name)
        elif report_name == 'somatic':
            self.somatic_markers.to_csv (sample_name + '_' + report_name + '.tsv', sep = '\t', index = False)


    def get_data (self) -> pd.DataFrame:
        """Return filtered marker data sorted by chromosome arm and position."""
        tmp = self.data
        return tmp.sort_values(by = ['arm', 'Pos'])


    def find_segments (self, skip_segmentation = False) -> None:
        """Segment allele imbalance after coverage and VAF filtering.

        The filtered markers are stored in ``self.filtered_data`` and the
        resulting :class:`Segments.Segment` object in ``self.segments``.
        ``skip_segmentation`` uses chromosome arms as the segment boundaries.
        """
        base_cvthr = float(self.config['AC_FILTER']['cvthr'])
        ptmthr = float(self.config['AC_FILTER']['PTMthr'])

        if self.adaptive_cvthr:
            adaptive_cvthrs = self.adaptive_threshold(self.data['ai'].values, base_cvthr)
            self.filtered_data = self.data.loc[(self.data['vaf'] >= ptmthr)&\
                                               (self.data['cv'] >= adaptive_cvthrs)]
            self.logger.info(f'Applied adaptive cvthr. Filtered to {len(self.filtered_data)} records.')
        else:
            self.filtered_data = self.data.loc[(self.data['vaf'] >= ptmthr)&\
                                               (self.data['cv'] >= base_cvthr)]
            self.logger.info(f'Applied fixed cvthr from config. Filtered to {len(self.filtered_data)} records.')
        
        self.segments = Segments.Segment(data = self.filtered_data.sort_values (by = ['Chr', 'Pos']),
                                         logger = self.logger, 
                                         msnfun = self.msnfun,
                                         signal_column = 'ai',
                                         config = self.config,
                                         include_HO = self.include_HO,
                                         merge_thr = float(self.config['SEGMENTATION']['ac_thr']),
                                         abs_thr = 0.25/np.sqrt(self.median_cv),
                                         f0 = float(self.config['SEGMENTATION']['ac_p0']), 
                                         length = int(self.config['SEGMENTATION']['ac_length']), 
                                         ms = float(self.config['SEGMENTATION']['ac_ms']),
                                         skip_segmentation = skip_segmentation) 
        

    def adaptive_threshold (self, ai, cvthr):
        """Return coverage thresholds adjusted for allele imbalance values."""
        new_cvthr = (1-4*ai**2)*cvthr
        new_cvthr[new_cvthr < cvthr/2] = cvthr/2
        return new_cvthr


    def get_segments (self):
        """Return the allele-imbalance segment table."""
        return self.segments.get_segments()


    def get_params (self, arm, start, end):
        """Return robust allele-imbalance parameters for ``[start, end)``.

        Returns ``(numpy.nan, numpy.nan)`` when the interval has no filtered
        markers.
        """
        tmp = self.filtered_data.loc[(self.data['arm'] == arm)] 
        tmp = tmp.loc[(tmp['Pos'] >= start) & (tmp['Pos'] < end)]
        if len(tmp):
            m, s, _ = self.msnfun(tmp)
            return (m, s)
        else:
            return (np.nan, np.nan)


    def msnfun (self, x):
        """Estimate the allele-imbalance median, spread, and marker count."""
        if len (x) < 10:
            return np.nan, np.nan, len(x)
        
        signal_column = 'ai'
        cv = x['cv'].values
        ac = x[signal_column].values
        n = len(ac)
        try:
            val, cts = np.unique (cv, return_counts = True)
            popt, pcov = opt.curve_fit (sts.norm.cdf, val, np.cumsum(cts)/np.sum(cts), 
                                        p0 = [np.median(cv), np.sqrt(np.median(cv))],
                                        bounds = ((np.quantile(cv, 0.1), 0.5*np.sqrt(np.median(cv))),
                                                (np.quantile(cv, 0.9), 3*np.sqrt(np.median(cv)))),
                                        sigma = np.sqrt(cts))
        except:
            popt = (np.median(cv),)
        if popt[0] < 1:
            popt[0] = np.median (cv)

        minstd = 1.1*np.sqrt(0.25/popt[0])
        maxstd = 3*np.sqrt(0.25/popt[0])
        std_0 = 1.1*np.sqrt(0.25/popt[0])
            
        val, cts = np.unique (ac, return_counts = True)
        
        if val[0] != 0:
            val = np.concatenate ((np.array((0,)), val))
            cts = np.concatenate ((np.array((0,)), cts))
        
        try:
            popt, pcov = opt.curve_fit (msn, val, np.cumsum(cts)/np.sum(cts), 
                                        p0 = [0.1,std_0 ,0.001,0.5],
                                        sigma = np.sqrt(cts+1),
                                        bounds = ((0.0001, minstd,0,  0.1),
                                                  (0.5,  maxstd, 0.3,0.999)))
            
            m, s, er, ho = popt
            sm, ss, ser, sho = np.sqrt(np.diag(pcov))

            he = (1 - er - ho)
            if he < 0.3:
                m = 0.5
                s = maxstd
            
            if (popt[1] >= maxstd) or (popt[1] <= minstd):
                self.logger.error (f'Distribution of ai is unusual. Model failed')
                m = np.nan
                s = np.nan
            
            return m, s, n
        
        except (TypeError, ValueError):
            return np.median(ac), np.quantile(ac, 0.5+0.341) - np.quantile(ac, 0.5-0.341), n 
            
            

def msn (v, m, s, noise, ho):
    """Model of allele-imbalance."""
    lin = 2*v
    cdf = _trunc_norm (v, m, s)
    res = noise*lin + (1-noise-ho)*cdf
    res[-1] += ho
    return res


def _trunc_norm (x, m, s, a = 0, b = 0.5):
    return (sts.norm.cdf(x, m, s) - sts.norm.cdf(a, m, s))/(sts.norm.cdf(b, m, s) - sts.norm.cdf(a, m, s))
