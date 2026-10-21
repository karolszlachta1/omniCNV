#!/usr/bin/env python3
"""
Re-draw omniCNV clone plots from saved TSVs (allele_counts, gene_counts, solution gen_segs)
without re-running segmentation or karyotyping.

Imports ``omniCNV.Genome`` from the installed package, or from a checkout (auto-detected from
this file’s location under ``helperScripts/`` in the repo, or via ``--rai-root`` / ``$RAI_ROOT``).

Requirements:
  - The same Python environment as the original omniCNV run (``omniCNV`` importable).
  - **Required:** ``-c`` / ``--config`` — the same INI as the original run (or equivalent
    ``[FILES]`` paths on this machine). No default is assumed.
  - matplotlib>=3.4 (Genome.plot_clones uses Figure.add_subfigure).

Example:

  python helperScripts/replay_plot_clones.py /path/to/sample_prefix -c /path/to/your_config.ini

See ``helperScripts/README.md`` for a full example command. Logging matches the main
``omniCNV`` tool (``UNPOP`` hierarchy; file ``{prefix}_replay_plot_clones.log`` next to outputs).

m0, scale, and d are read only from the log *section* for this sample: the script finds
run blocks starting with "Running omniCNV with the following parameters:" and picks the latest
block whose `sample_name:` matches the replay prefix (path or basename), so concatenated
logs or multiple runs do not mix values across samples or old runs.

Optimization distance "d" is NOT stored in any TSV. When replaying, d comes from
"Optimization distance (d):" lines in that same section, or from --distance.

Older logs without d lines: use --distance or re-run omniCNV once to refresh the .log file.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple


def _configure_replay_logger(prefix: Path, verbosity: str) -> logging.Logger:
    """
    Match rAI ``__main__.get_logger`` style: root ``UNPOP``, same formatter, file + stderr.

    Log file: ``{prefix.parent}/{prefix.name}_replay_plot_clones.log``
    Child logger name: ``UNPOP.replay_plot_clones`` (same dotted style as ``UNPOP.Genome.Solution``).
    """
    root = logging.getLogger("UNPOP")
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s %(name)s: %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log_file = prefix.parent / f"{prefix.name}_replay_plot_clones.log"
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(getattr(logging, verbosity))
    sh.setFormatter(fmt)
    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(sh)
    return root.getChild("replay_plot_clones")


def _require_matplotlib_version() -> None:
    """Genome.plot_clones uses Figure.add_subfigure (matplotlib >= 3.4)."""
    import matplotlib

    parts = matplotlib.__version__.split(".")
    try:
        major, minor = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return
    if (major, minor) < (3, 4):
        print(
            f"replay_plot_clones requires matplotlib>=3.4 (need Figure.add_subfigure). "
            f"Current: {matplotlib.__version__}. Example: pip install 'matplotlib>=3.4'",
            file=sys.stderr,
        )
        sys.exit(1)


def _patch_matplotlib_set_xticks_rotation() -> None:
    """
    Genome.plot_per_chromosome uses set_xticks(pos, labels, rotation=90).
    Matplotlib versions before ~3.8 do not support rotation= there; apply labels with set_xticklabels.
    """
    try:
        import matplotlib.axes._base as mbase
    except ImportError:
        return

    cls = mbase._AxesBase
    if getattr(cls.set_xticks, "_rAI_replay_patched", False):
        return
    _orig = cls.set_xticks

    def set_xticks(self, *args, **kwargs):
        rotation = kwargs.pop("rotation", None)
        try:
            return _orig(self, *args, **kwargs)
        except TypeError:
            if rotation is None or len(args) < 2:
                raise
            ticks, labels = args[0], args[1]
            _orig(self, ticks, **kwargs)
            self.set_xticklabels(labels, rotation=rotation)
            return

    set_xticks._rAI_replay_patched = True  # type: ignore[attr-defined]
    cls.set_xticks = set_xticks


def _find_rai_repo_root_from_script() -> Optional[Path]:
    """
    If this file lives under an rAIg checkout (e.g. rAIg/helperScripts/replay_plot_clones.py),
    return the repo root (directory that contains rAI/__init__.py).
    """
    here = Path(__file__).resolve().parent
    for _ in range(8):
        if (here / "rAI" / "__init__.py").is_file():
            return here
        if here.parent == here:
            break
        here = here.parent
    return None


def _ensure_rai_on_path(rai_root: Optional[Path]) -> None:
    """Import rAI from an installed package or by prepending rAIg repo root to sys.path."""
    try:
        import omniCNV  # noqa: F401

        return
    except ImportError:
        pass

    candidates: List[Path] = []
    if rai_root is not None:
        candidates.append(rai_root.resolve())
    env = os.environ.get("RAI_ROOT")
    if env:
        candidates.append(Path(env).resolve())
    discovered = _find_rai_repo_root_from_script()
    if discovered is not None:
        candidates.append(discovered)

    for root in candidates:
        init = root / "rAI" / "__init__.py"
        if init.is_file():
            sys.path.insert(0, str(root))
            import omniCNV  # noqa: F401

            return
    raise ImportError(
        "Could not import rAI. Install the package (pip install -e /path/to/rAIg), set "
        "RAI_ROOT to the repository root, or pass --rai-root pointing to the directory "
        "that contains the rAI/ package."
    )


RUN_START_MARKER = "Running rAI with the following parameters:"

_RE_OPT = re.compile(
    r"Optimized solution with m0:\s*([0-9.eE+-]+)\s*,\s*scale:\s*([0-9.eE+-]+)",
    re.MULTILINE,
)

# Logged in Solution.optimize() after self.distance = result.fun
_RE_LOG_DISTANCE = re.compile(
    r"Optimization distance \(d\):\s*(nan|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
    re.MULTILINE,
)

_RE_LOG_SAMPLE_NAME = re.compile(r"sample_name:\s*(.+)", re.MULTILINE)


def _parse_float_token(s: str) -> float:
    """Log lines may end the scale with a sentence period; strip it before float()."""
    return float(s.rstrip("."))


def _split_log_runs(text: str) -> List[str]:
    """Split log text into blocks, one per rAI invocation (each starts with RUN_START_MARKER)."""
    positions = [m.start() for m in re.finditer(re.escape(RUN_START_MARKER), text)]
    if not positions:
        return [text]
    runs: List[str] = []
    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        runs.append(text[pos:end])
    return runs


def _parse_sample_name_from_run(run_text: str) -> Optional[str]:
    m = _RE_LOG_SAMPLE_NAME.search(run_text)
    if not m:
        return None
    return m.group(1).strip()


def _prefix_matches_log_sample(prefix: Path, sample_from_log: str) -> bool:
    """True if replay prefix refers to the same output as rAI's logged sample_name."""
    s = sample_from_log.strip()
    if not s:
        return False
    try:
        if Path(s).expanduser().resolve() == prefix.expanduser().resolve():
            return True
    except Exception:
        pass
    try:
        if os.path.normpath(os.path.abspath(s)) == os.path.normpath(os.path.abspath(str(prefix))):
            return True
    except Exception:
        pass
    return Path(s).name == prefix.name


def _select_log_run_for_prefix(
    full_text: str, prefix: Path, logger: logging.Logger
) -> str:
    """
    Return only the log slice for this sample.

    Prefers the *latest* run block whose `sample_name:` matches the replay prefix (same
    path or same basename). If the log was concatenated or contains multiple samples,
    this avoids mixing m0/scale/d from another sample or an earlier run.

    If no block matches, uses the *last* run block and logs a warning.
    """
    runs = _split_log_runs(full_text)
    if len(runs) == 1 and RUN_START_MARKER not in runs[0]:
        logger.debug("Log has no %r marker; parsing entire file.", RUN_START_MARKER)
        return runs[0]

    for run in reversed(runs):
        logged = _parse_sample_name_from_run(run)
        if logged and _prefix_matches_log_sample(prefix, logged):
            logger.info(
                "Using log section for sample_name=%s (matches replay prefix %s).",
                logged,
                prefix,
            )
            return run

    if runs:
        logger.warning(
            "No log section matched replay prefix %s (check sample_name lines). "
            "Using only the LAST run block in the file — d/m0/scale may be wrong if this "
            "log mixes samples or runs.",
            prefix,
        )
        return runs[-1]
    return full_text


def _parse_log_solutions_from_text(text: str) -> List[Tuple[float, float]]:
    """Return list of (m0, scale) in order (one per 'Optimized solution' line in this text)."""
    return [
        (_parse_float_token(m.group(1)), _parse_float_token(m.group(2))) for m in _RE_OPT.finditer(text)
    ]


def _parse_log_distances_from_text(text: str) -> List[float]:
    """
    Return list of d values in order of appearance (one per 'Optimization distance (d):' line).
    """
    out: List[float] = []
    for m in _RE_LOG_DISTANCE.finditer(text):
        tok = m.group(1).lower()
        if tok == "nan":
            out.append(float("nan"))
        else:
            out.append(float(tok))
    return out


class _LoadedSolution:
    """Enough of rAI.Solution for Genome.plot_clones and helpers."""

    def __init__(self, segments_df, m0: float, scale: float, distance: float):
        self._segments = segments_df
        self.m0 = m0
        self.scale = scale
        self.distance = distance

    def get_segments(self):
        return self._segments


class ReplayGenome:
    """
    Same plotting API as Genome.Genome for plot_clones, without merge/karyotype.

    We subclass after rAI import so methods resolve to Genome.Genome.
    """

    def __init__(self, genome_cls, config, logger, allele_counts, gene_counts, solution, coverage_type: str):
        self.config = config
        self.logger = logger
        self.gene_counts = gene_counts
        self.allele_counts = allele_counts
        self.coverage_type = coverage_type
        self.segments = solution
        self.m0 = solution.m0
        self.scale = solution.scale
        self.distance = getattr(solution, "distance", float("nan"))
        # Bind plotting methods from Genome (no __init__ call on Genome)
        for name in ("plot_clones", "plot_karyotypes", "plot_per_chromosome", "plot_karyotypes_chrom"):
            setattr(self, name, genome_cls.__dict__[name].__get__(self, ReplayGenome))


def _glob_gen_segs(prefix: Path) -> List[Path]:
    """Paths like prefix_solution_1_gen_segs.tsv sorted by solution index."""
    parent = prefix.parent
    stem = prefix.name
    pattern = f"{stem}_solution_*_gen_segs.tsv"
    paths = sorted(parent.glob(pattern), key=lambda p: int(re.search(r"_solution_(\d+)_", p.name).group(1)))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay rAI plot_clones from saved TSVs.")
    parser.add_argument(
        "prefix",
        type=str,
        help="Output prefix used by rAI (e.g. /path/to/test_v01 — same as -s).",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        metavar="INI",
        help=(
            "Config INI for this replay (same as the original rAI ``-c``): FILES.arms, "
            "FILES.cyto, etc. Must be supplied explicitly; no default."
        ),
    )
    parser.add_argument(
        "-t",
        "--type-coverage",
        default="atac",
        choices=["rna", "wes", "atac", "wgs"],
        help="Must match the original -t (default: atac).",
    )
    parser.add_argument(
        "--rai-root",
        type=str,
        default=None,
        help=(
            "Root of the rAIg repo (directory that contains rAI/). Usually unnecessary: "
            "use an installed rAI, or run this script from a checkout (auto-detected), "
            "or set RAI_ROOT."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Run log to parse m0, scale, and (if present) d. Default: {prefix}.log next to outputs.",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=None,
        help=(
            "Override optimization distance d for the subtitle. If omitted, the script reads "
            "'Optimization distance (d):' from the log when present; otherwise NaN."
        ),
    )
    parser.add_argument(
        "--logging-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    _ensure_rai_on_path(Path(args.rai_root) if args.rai_root else None)

    import configparser
    import pandas as pd

    from omniCNV.Genome import Genome as GenomeCls

    _require_matplotlib_version()
    _patch_matplotlib_set_xticks_rotation()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    prefix = Path(args.prefix).expanduser().resolve()
    log_path = Path(args.log_file).expanduser().resolve() if args.log_file else (prefix.parent / f"{prefix.name}.log")
    ac_path = prefix.parent / f"{prefix.name}_allele_counts.tsv"
    gc_path = prefix.parent / f"{prefix.name}_gene_counts.tsv"

    for p in (ac_path, gc_path):
        if not p.is_file():
            print(f"Missing required file: {p}", file=sys.stderr)
            return 1

    config = configparser.ConfigParser()
    config.read(config_path)

    logger = _configure_replay_logger(prefix, args.logging_level)
    logger.info("Using config: %s", config_path)

    ac_df = pd.read_csv(ac_path, sep="\t")
    gc_df = pd.read_csv(gc_path, sep="\t")
    allele_counts = SimpleNamespace(data=ac_df)
    gene_counts = SimpleNamespace(data=gc_df)

    gen_seg_paths = _glob_gen_segs(prefix)
    if not gen_seg_paths:
        print(
            f"No files matching {prefix.name}_solution_*_gen_segs.tsv under {prefix.parent}",
            file=sys.stderr,
        )
        return 1

    log_pairs: List[Tuple[float, float]] = []
    log_distances: List[float] = []
    if log_path.is_file():
        full_log = log_path.read_text(encoding="utf-8", errors="replace")
        run_text = _select_log_run_for_prefix(full_log, prefix, logger)
        log_pairs = _parse_log_solutions_from_text(run_text)
        log_distances = _parse_log_distances_from_text(run_text)
        if len(log_distances) != len(log_pairs) and log_distances and log_pairs:
            logger.warning(
                "In the selected log section, %d 'd' lines vs %d m0/scale lines — "
                "pairing by solution index; verify your log.",
                len(log_distances),
                len(log_pairs),
            )
    if args.distance is None and not log_distances and log_path.is_file():
        logger.warning(
            "No 'Optimization distance (d):' lines in the selected log section of %s — subtitle d "
            "will be NaN unless you use --distance. Re-run rAI with current code to log d.",
            log_path,
        )

    for i, gpath in enumerate(gen_seg_paths):
        seg_df = pd.read_csv(gpath, sep="\t")
        if i < len(log_pairs):
            m0, scale = log_pairs[i]
        elif len(log_pairs) == 1:
            m0, scale = log_pairs[0]
        else:
            print(
                f"Could not find m0/scale in {log_path} for solution index {i + 1}. "
                "Add matching 'Optimized solution' lines or extend this script.",
                file=sys.stderr,
            )
            return 1

        if args.distance is not None:
            dist = args.distance
        elif i < len(log_distances):
            dist = log_distances[i]
        elif len(log_distances) == 1:
            dist = log_distances[0]
        else:
            dist = float("nan")

        sol = _LoadedSolution(seg_df, m0=m0, scale=scale, distance=dist)
        replay = ReplayGenome(GenomeCls, config, logger, allele_counts, gene_counts, sol, args.type_coverage)
        out_base = str(prefix) + f"_solution_{i + 1}"
        logger.info("Writing %s_clones.png (m0=%s scale=%s d=%s)", out_base, m0, scale, dist)
        replay.plot_clones(out_base)

    return 0


if __name__ == "__main__":
    sys.exit(main())
