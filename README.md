# omniCNV package

The tool meant for CNV analysis of (almost) any sequencing data.

## Installing omniCNV

# Clone the repository
```shell
git clone https://github.com/karolszlachta1/omniCNV
```

### Using conda
```shell
# create env with python 3.11+
conda create -n omniCNV_env python=3.11 -y

# activate env
conda activate omniCNV_env

# install omniCNV itself (also after each code change - reinstall it each time)
cd /path/to/rAIg
pip install .
# or `pip install -e .` for development

# verify
omniCNV -h
```

### Using python venv
```shell
# setup a virtual environment
python -m venv .venv
# activate the virtual environment
source ./.venv/bin/activate

# install into virtual environment
pip install .

# upgrade pip and clean cache in case of errors
pip install --upgrade pip
rm -rf ~/.cache/pip
```

### Running tests

Install the package with its test dependencies, then run the full test suite:

```shell
pip install -e ".[test]"
python -m pytest -q
```

The CI workflow runs the same install and test commands. Large synthetic datasets
are generated locally and are intentionally excluded from version control.

### Synthetic WGS data

The synthetic generator uses hg38 chromosome sizes, random marker placement, and
simulated CNV truth records. Generate a local dataset with:

```shell
python tests/generate_synthetic_dataset.py \
  --output-dir tests/fixtures/generated_latest \
  --prefix hg38
```

The generated directory contains the VCF, allele-count TSV, marker list, arms,
cytoband reference, configuration, and `cnv_truth.tsv` for comparison with
segmentation results.

See [tests/README.md](tests/README.md) for the complete synthetic-data workflow
and, importantly, the scope of what these tests do and *DO NOT* validate. Synthetic
simulation results are runtime/file-format checks only; they are not evidence of
scientific or CNV-detection correctness.

## Running omniCNV

### Configuration template

Start from [omniCNV/config_template.ini](omniCNV/config_template.ini). It lists
the required sections, portable path placeholders, and recommended starting values
for various omics. Copy it to a new config file,
replace the reference paths, and adjust the input column names for the selected
modality.

Depending on input data type, slightly different inputs are required.
Here are minimal examples.

### Requirements

RAM: 30 GB for WGS 
     10 GB for WES,RNA,ATAC,CHIP 
CPU: 1 

### WGS
Using an allele-count TSV:
```shell
omniCNV -t wgs \
        -c config.ini \
        -s path/sample_name \
        -ac allele_counts.tsv
```

Using a VCF:
```shell
omniCNV -t wgs \
        -c config.ini \
        -s path/sample_name \
        -v sample.vcf
```

### RNA/WES/ATAC/CHIP
For RNA, using an allele-count TSV:
```shell
omniCNV -t rna \
        -c config.ini \
        -s path/sample_name \
        -rp reference_panel.tsv \
  -ac allele_counts.tsv \
        -gc gene_counts.tsv
```

Using a VCF instead:
```shell
omniCNV -t rna \
        -c config.ini \
        -s path/sample_name \
        -rp reference_panel.tsv \
        -v sample.vcf \
        -gc gene_counts.tsv
```

Replace `rna` with `wes`, `atac`, or `chip` as needed.

## Full set of parameters
For the complete and current list of options, run:
```shell
omniCNV -h
```

The main options are documented below:

- -gc, --gene_counts GENE_COUNTS
                        Gene counts file
- -t, --type_coverage {rna,wes,atac,wgs,chip}
                        Type of input data.
- -rp, --reference_panel REFERENCE_PANEL
                        Reference panel for gene counts
- -c, --config CONFIG
                        INI file with configuration. Default: config.ini
- -s, --sample_name SAMPLE_NAME
                        Core name to be used for output files
- -r, --reports [{allele_counts,gene_counts,plot_clones,reference,gc_segs,ac_segs,ambiguous_regions} ...]
                        Reports to report. Default: allele_counts, gene_counts, plot_clones, reference, gc_segs, ac_segs, ambiguous_regions.

Allele counts:
  File containing allele counts

- -ac, --allele_counts ALLELE_COUNTS
                        File in tsv format
- -v, --vcf VCF     File in vcf format


## Input data formats

### Allele counts 

Allele counts can be extracted from vcf or gvcf file (with -v, --vcf argument) as long as fields as specified in [VCF_FORMAT] section of config file are present in vcf file and correctly configured.
Another option is to supply tab separated file with columns as specified in [ALLELIC_COUNT_FORMAT] in config file.

### Reference panel

Reference panel is a tab separated file with columns:
- geneID: gene or region identifier; values must match the gene IDs in the sample gene-count file
- arm: chromosome arm, for example `chr1p` or `chr1q`
- refcpm: mean reference coverage for the gene or region, in counts per million
- Pos: genomic position assigned to the gene or region

### Gene counts

Gene counts is tab separated file with two columns: 
- geneID - values must match values of geneID in reference panel
- count - coverage

## Outputs

omniCNV has ability to generate multiple reports:
- allele_counts: list of filtered markers used for allelic imbalance assesment 
- chrY_allele_counts: list of filtered markers used for chrY assesment
- gene_counts: gene counts as used for analysis
- plot_clones: comprehensive visualisation of the results
- reference: boxplot showing coverage per arm
- gc_segs: segments as determined based on gene counts
- ac_segs: segments as determined based on allelic imbalance
- ambiguous_regions: bed file with ambiguous regions, that is regions inbetween segment that cannot have confidently determined genomic parameters 

Copyright (c) 2026 St. Jude Children's Research Hospital
