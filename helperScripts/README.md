# Helper scripts

## `replay_plot_clones.py`

Re-draws omniCNV **clone plots** (`*_clones.png`) from saved outputs **without** re-running segmentation or karyotyping. It reads:

- `{prefix}_allele_counts.tsv` and `{prefix}_gene_counts.tsv`
- `{prefix}_solution_*_gen_segs.tsv`
- The original run log (default: `{prefix}.log`) for `m0`, `scale`, and optional `d`

You must pass **`-c`** with the same config INI used for the original omniCNV run (or an equivalent INI whose `[FILES]` paths resolve on this machine).

### Example

```bash
conda activate omniCNV_env
sample="test_v02"
python /path/to/omniCNV/helperScripts/replay_plot_clones.py ${sample} -c /path/to/your_config.ini
```

Use the directory and INI paths that match **your** installation. The `sample` value is the omniCNV output **prefix** (same as `-s` / `--sample_name`).

Logs are written in the same style as the main `omniCNV` tool to stderr and to `{prefix}_replay_plot_clones.log` next to the outputs.

See the script docstring for options (`-t`, `--log-file`, `--distance`, etc.).


## `calculate reference panel`

calc_ref_panel.py -h
usage: calc_ref_panel.py [-h] -f FILE_LIST -a ANNOTATION [-rp REFERENCE_PANEL] [-x] [-c CPM_THR] [-s NO_SAMPLES] [-l LOG_LEVEL]

Build male and female reference panels by merging input files on geneID and attaching annotation.

options:
  -h, --help            show this help message and exit
  -f FILE_LIST, --file-list FILE_LIST
                        Text file with female input file paths, one per line.
  -a ANNOTATION, --annotation ANNOTATION
                        Annotation table with geneID column.
  -rp REFERENCE_PANEL, --reference_panel REFERENCE_PANEL
                        Output file for panel. Default: reference_panel.tsv
  -x, --correct-x       Whether to apply 2x correction to chrX counts in male samples. Default: False.
  -c CPM_THR, --cpm-thr CPM_THR
                        Minimum CPM threshold for gene inclusion. Default: 1.
  -s NO_SAMPLES, --no_samples NO_SAMPLES
                        Minimum number of samples with CPM above threshold to include gene. Default: 5.
  -l LOG_LEVEL, --log-level LOG_LEVEL
                        Logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO.

Input files are expected to have at least two columns: gene_id, and expected_count.
Annotation file is expected to have at least ... columns: geneID, Pos, arm.

Note that any additional filtering can be done on annotation file, and will be automatically applied to reference panel. 

