import os

import argparse
import sys
import configparser
import logging
from importlib.metadata import version, PackageNotFoundError

from omniCNV import AlleleCounts as AlleleCounts
from omniCNV import GeneCounts as GeneCounts
from omniCNV import Genome as Genome
from omniCNV import Reports as Reports

def main ():
    """Parse CLI options, run the omniCNV workflow, and generate reports.

    Input validation and modality-specific option handling occur before allele
    counts, gene counts, genome integration, karyotyping, and report dispatch.
    Returns zero after successful completion.
    """

    parser = argparse.ArgumentParser(description = 'omniCNV: A tool for analyzing sequencing data for CNV.')
    parser.add_argument('-V', '--version', action='version',
                        version=f'%(prog)s {version("omniCNV")}')
    parser.add_argument ('-gc', '--gene_counts', required = False, help = 'Gene counts file')
    parser.add_argument ('-t', '--type_coverage', required = True, help = 'Type of input data.',
                         choices = ['rna', 'wes', 'atac','wgs','chip'])
    parser.add_argument ('-rp', '--reference_panel', required = False, help = 'Reference panel for gene counts')

    group = parser.add_argument_group ('Allele counts', "File containing allele counts")
    exclusive_group = group.add_mutually_exclusive_group (required = True)
    exclusive_group.add_argument ('-ac', '--allele_counts', help = 'File in tsv format')
    exclusive_group.add_argument ('-v', '--vcf', help = 'File in vcf format')

    parser.add_argument ('-c', '--config', default = os.path.dirname(__file__)+'/config.ini', type = open,
                         help = 'INI file with configuration. Default: config.ini')
    parser.add_argument ('-s', '--sample_name', help = 'Core name to be used for output files',
                         default = '')
    parser.add_argument ('-r', '--reports', help = f'Reports to report. Default: {", ".join(Reports.Report.get_reports_list())}.', 
                         choices = Reports.Report.get_reports_list(), nargs = '*', action = 'extend',
                         default = Reports.Report.get_reports_list())
    parser.add_argument ('-l', '--logging_level', help = 'Logger verbosity level, default: WARNING',
                         default = 'WARNING', choices = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])

    advanced_group = parser.add_argument_group ('ADVANCED',
                                                "Advanced and or experimental parameters. Please use with CAUSION and make sure you know what you're doing.")
    advanced_group.add_argument ('--adaptive_cvthr', action = 'store_true',
                                 help = 'Use AI-adaptive coverage threshold for allele-count segmentation (default: disabled).')
    #TODO: This option needs to be improved. Namely remove fraction of HO only to keep vaf = 1 from cnv.
    advanced_group.add_argument ('--include_HO', action = 'store_true',
                                 help = 'Include vaf = 1 markers in ai segmentation (default: disabled). Exp')
    advanced_group.add_argument ('--skip_segmentation', action = 'store_true',
                                 help = 'Skip segmentation step and use chromosome arms (default: disabled).')
    advanced_group.add_argument ('--apply_log_to_coverage', action = 'store_true',
                                 help = 'Apply log transformation to coverage data (default: disabled).')
    advanced_group.add_argument ('--analyze_chrY', action = 'store_true',
                                 help = 'Analyze chromosome Y (default: disabled). Only for WGS data.')
    advanced_group.add_argument ('--skip_filter_good_SNP', action = 'store_true',
                                 help = 'Skip filtering of good SNPs, WGS only (default: disabled, i.e. filtering not performed).')

    args = parser.parse_args()

    logger = get_logger (args.sample_name, args.logging_level)
    logger.info (f'Running omniCNV with the following parameters:')
    for key,value in args.__dict__.items():
        logger.info (f'{key}: {value}')

    if (args.type_coverage == 'wgs'):
        if (args.reference_panel is not None):
            logger.warning ('Reference panel is not used for WGS data. It will be ignored.')
            args.reference_panel = None
        if (args.gene_counts is not None):
            logger.warning ('Gene counts are not used for WGS data. It will be ignored.')
            args.gene_counts = None
        if ('gene_counts' in args.reports):
            logger.warning ('Gene counts report is not generated for WGS data. It will be ignored.')
            args.reports.remove('gene_counts')
            
    elif (args.type_coverage != 'wgs') and (args.reference_panel is None):
        logger.error ('Reference panel is required for RNA, WES and ATAC data.')
        logger.error ('Please provide a reference panel.')
        logger.error ('Please ...')
        sys.exit (1)
    
    if (args.analyze_chrY) and (args.type_coverage != 'wgs'):
        logger.error ('Chromosome Y analysis is only available for WGS data. Ignoring --analyze_chrY flag.')
        args.analyze_chrY = False

    if (args.skip_filter_good_SNP) and (args.type_coverage != 'wgs'):
        logger.error ('Skipping filtering of good SNPs is only available for WGS data. Ignoring --skip_filter_good_SNP flag.')
        args.skip_filter_good_SNP = False

    if args.sample_name is None:
        # gene_counts may be None for WGS; fall back to allele counts path
        source = args.gene_counts or args.allele_counts or args.vcf
        args.sample_name = os.path.basename(source).split('.')[0] if source else 'sample'

    if args.sample_name == '':
        args.sample_name = 'sample'

    config = configparser.ConfigParser ()
    config.read_file (args.config)

    if args.allele_counts is not None:
        vaf_input = 'allele_counts'
        data_file = args.allele_counts
    else:
        vaf_input = 'vcf'
        data_file = args.vcf

    ac = AlleleCounts.AlleleCounts (data_file = data_file,
                                    input_type = vaf_input,
                                    config = config,
                                    logger = logger,
                                    adaptive_cvthr = args.adaptive_cvthr,
                                    include_HO = args.include_HO,
                                    skip_segmentation = args.skip_segmentation,
                                    analyze_chrY = args.analyze_chrY,
                                    skip_filter_good_SNP = args.skip_filter_good_SNP)

    if args.gene_counts is not None:
        df = args.gene_counts
    else:
        df = ac.data
    gc = GeneCounts.GeneCounts (data_file = df, 
                                reference_panel = args.reference_panel,
                                data_type = args.type_coverage,
                                config = config,
                                logger = logger,
                                skip_segmentation = args.skip_segmentation,
                                apply_log_to_coverage = args.apply_log_to_coverage)
    
    genome = Genome.Genome (gene_counts = gc,
                            allele_counts = ac,
                            coverage_type = args.type_coverage,
                            config = config,
                            logger = logger,
                            sample_name = args.sample_name)

    genome.karyotype(models = ['A', 'AA', 'AAB', 'AAA']) #, 'AAA'

    report = Reports.Report (genome = genome, 
                             allele_counts = ac,
                             gene_counts = gc,
                             config = config,
                             logger = logger)
    
    reports = args.reports
    if args.analyze_chrY:
        reports.append('chrY')
    report.generate (reports = args.reports, 
                     sample_name = args.sample_name)
    
    return 0

def get_logger (sample, verbosity = 'DEBUG') -> logging.Logger:
    """Create the file and console log handlers for a sample run."""
    logger = logging.getLogger ('omniCNV')
    logger.setLevel(verbosity)
    fh = logging.FileHandler(sample + '.log', mode = 'w')
    fh_formatter = logging.Formatter('%(asctime)s %(name)s: %(levelname)s: %(message)s',
                                     datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler ()
    sh.setLevel (verbosity)
    sh.setFormatter (fh_formatter)
    logger.addHandler (sh)
    return logger

if __name__ == '__main__':
    if not sys.warnoptions:
        import warnings
    warnings.simplefilter("ignore")
    sys.exit(main())
