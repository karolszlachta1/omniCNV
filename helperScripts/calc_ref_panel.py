import numpy as np
import pandas as pd 
import argparse
import logging
from pathlib import Path



TMM_Acut = 0.05
TMM_Mcut = 0.3

def TMM (ref, sampl, Acut = TMM_Acut, Mcut = TMM_Mcut):
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


def read_inputs (list_file, logger):
    
    with open(list_file, 'r') as f:
        lines = f.readlines()
    
    male = [] #pd.DataFrame(columns = ['gene_id'])
    female = [] #pd.DataFrame(columns = ['gene_id'])

    for line in lines:
        file, sex = line.strip().split ('\t') 
        data = pd.read_csv (file, sep = '\t')
        sid = file.split('/')[-1].split('.')[0]
        logger.debug(f"Reading sample {sid}.")

        data.rename (columns = {'expected_count': str(sid)}, inplace = True)
        #print (data.head())
        if sex == 'M':
            if len(male):
                male = male.merge(data[['geneID', str(sid)]], how = 'outer', on = 'geneID')
                #print ('i: ', male.head())
            else:
                male = data[['geneID', str(sid)]].copy()
                #print (male.head())
        elif sex == 'F':
            if len(female):
                female = female.merge(data[['geneID', str(sid)]], how = 'outer', on = 'geneID')
                #print ('i: ', female.head())
            else:
                female = data[['geneID', str(sid)]].copy()
                #print (female.head())
    #print (male.head())

    male.set_index ('geneID', drop = True, inplace = True)
    female.set_index ('geneID', drop = True, inplace = True)

    logger.info(f"""Finished reading input files for:
                 {len(male.columns)} male and {len(female.columns)} female samples.""")

    return male, female

def correct_chrX (male, chrX_genes, logger):
    tmp = male.loc[[g in chrX_genes for g in male.index.values],:]

    male.loc[tmp.index, ] = 2*tmp 
    
    logger.info(f"Applied 2x correction to {len(tmp)} genes on chrX in male samples.")
    return male


def tmm_normalize_counts (df, logger):
    
    for col in df.columns[1:]:
        f = TMM (df[df.columns[0]].values, df[col].values)
        logger.info (f'Sample {col} vs sample {df.columns[0]} TMM: {f}')
        df[col] = df[col]*2**f

    return df


def build_reference_panels(list_file: str,
                           annotation_file: str,
                           out: str,
                           x_correction: bool,
                           cpm: float,
                           no_samples: int,
                           logger) -> int:
    
    male, female = read_inputs (list_file, logger)

    annot = pd.read_csv (annotation_file, sep = '\t')
    if x_correction:
        logger.info("Applying 2x correction to chrX counts in male samples.")
        male = correct_chrX (male, annot.loc [[x[:-1] == 'chrX' for x in annot['arm']], 'geneID'])

    cpms = tmm_normalize_counts (pd.merge (male, female, left_index = True, right_index = True),
                                 logger = logger)
    cpmfilt = (cpms > cpm).sum (axis = 1) >= no_samples
    logger.info (f"Filtering genes with CPM > {cpm} in at least {no_samples} samples.")
    logger.info (f" Remaining genes: {cpmfilt.sum()}.")
    cpms['refcpm'] = cpms.mean (axis = 1)
    cpms = cpms.loc[cpmfilt].copy()

    annotated = pd.merge (cpms[['refcpm']], annot,
                          on = 'geneID',    how = 'inner')
    annotated.to_csv (out, sep = '\t', index = False)
    return 0

def _setup_logger(log_level: str, log_file_name: str) -> logging.Logger:
    """Configure logger to write to both file and stderr."""
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    
    # File handler
    fh = logging.FileHandler(log_file_name, mode='w')
    fh.setLevel(logging.DEBUG)
    
    # Stderr handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(sh)
    
    return logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Build male and female reference panels by merging input files on geneID "
                     "and attaching annotation."))
    
    parser.add_argument("-f","--file-list", required = True,
                        help = "Text file with female input file paths, one per line.")
    parser.add_argument("-a", "--annotation", required = True,
                        help = "Annotation table with geneID column.")
    parser.add_argument("-rp", "--reference_panel", default = "reference_panel.tsv",
                        help = "Output file for panel. Default: reference_panel.tsv")
    parser.add_argument("-x", "--correct-x", default = False, action = "store_true",
                        help = "Whether to apply 2x correction to chrX counts in male samples. Default: False.")
    parser.add_argument("-c", "--cpm-thr", default = 1, type = float,
                        help = "Minimum CPM threshold for gene inclusion. Default: 1.")
    parser.add_argument("-s", "--no_samples", default = 5, type = int,
                        help = "Minimum number of samples with CPM above threshold to include gene. Default: 5.")
    parser.add_argument("-l", "--log-level", default = "INFO",
                        help = "Logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO.")
    
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logger = _setup_logger(args.log_level, args.reference_panel.split('.')[0] + '.log')
    
    logger.info(f"Starting reference panel generation with parameters:")
    logger.debug(f"  File list: {args.file_list}")
    logger.debug(f"  Annotation file: {args.annotation}")
    logger.debug(f"  Output file: {args.reference_panel}")
    logger.debug(f"  Correct X chromosome: {args.correct_x}")
    logger.debug(f"  CPM threshold: {args.cpm_thr}")
    logger.debug(f"  Minimum samples: {args.no_samples}")

    # Check if input files
    files = []
    
    with open(args.file_list, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        file = line.split ('\t')[0] # strip()
        if Path(file).is_file():
            files.append(file)
            logger.debug(f"Found input file: {file}")
        else:
            logger.error(f"File {file} does not exist.")

    #Check number of valid input files
    if len(files) < args.no_samples:
        logger.error(f"Number of inputs: {len(files)} is less than required: {args.no_samples}.")
        return 1
    if len(files) <= 10:
        logger.warning(f"Only {len(files)} input files found. Consider providing more samples for a robust reference panel.")
    elif len(files) < 20:
        logger.info(f"{len(files)} input files found. This is a moderate number of samples for a reference panel.")

    #Check if annotation file exists    if not Path(args.annotation).is_file():
    if not Path(args.annotation).is_file():
        logger.error(f"Annotation file {args.annotation} does not exist.")
        return 1
    
    try:
        logger.info("Building reference panels...")
        result = build_reference_panels(list_file = args.file_list,
                                        annotation_file = args.annotation,
                                        out = args.reference_panel,
                                        x_correction = args.correct_x,
                                        cpm = args.cpm_thr,
                                        no_samples = args.no_samples,
                                        logger = logger)

        logger.info(f"Reference panel generation completed with exit code: {result}")
        return result
    except Exception as e:
        logger.error(f"Error during reference panel generation: {e}")
        return 1
    
if __name__ == "__main__":
    exit(main())