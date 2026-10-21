
REPORTS_LIST = ['allele_counts', 'gene_counts', 'plot_clones', 
                'reference', 'gc_segs', 'ac_segs', 'ambiguous_regions']
                #'somatic','ROH','karyotypes','gen_segs',, 'igv'

class Report:
    """
    Report class for generating various types of reports.
    Holds references to genome, allele counts, and gene counts objects.
    Provides a unified interface to generate reports for a given sample.
    """
    def __init__(self, genome, allele_counts, gene_counts, config, logger) -> None: 
        """Create a dispatcher for genome, allele-count, and gene-count reports."""
        self.genome = genome
        self.allele_counts = allele_counts
        self.gene_counts = gene_counts
        self.config = config
        self.logger = logger.getChild(f'{self.__class__.__name__}')

    def generate(self, reports, sample_name):
        """Generate each requested report using the owning analysis object.

        ``reports`` contains report-name strings from :data:`REPORTS_LIST`.
        Unsupported names raise ``ValueError``; ``sample_name`` is used as the
        output filename prefix.
        """
        
        for report_name in reports:
            if report_name in ['allele_counts', 'ac_segs', 'somatic', 'chrY']:
                self.allele_counts.generate_report(sample_name, report_name)
            elif report_name in ['gene_counts', 'reference', 'gc_segs']:
                self.gene_counts.generate_report(sample_name, report_name)
            elif report_name in ['plot_clones', 'ambiguous_regions', 'igv']:
                self.genome.generate_report(sample_name, report_name)
            elif report_name == 'chrY':
                self.chrY.generate_report(sample_name, report_name)
            else:
                raise ValueError(f"Unknown report type: {report_name}")

        
    @staticmethod
    def get_reports_list():
        """Return the report names supported by the command-line interface."""
        return REPORTS_LIST