"""Published benchmark pages moved here from torchfits."""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_benchmark_run_ids_match_published_assets() -> None:
    """Every run ID the benchmarks page cites must have published CSVs.

    The API-freeze skill's Phase 5 used to send a reviewer after a
    `BENCH_SNAPSHOT` marker in `docs/benchmarks.md` and a matching
    `benchmarks_results/<run-id>/` directory. The marker exists nowhere in the
    repository, `benchmarks_results/` is gitignored local scratch, and the README
    has no performance table at all -- so the phase could only ever end in "the
    thing you were asked to check is not here". The published CSVs are mirrored
    under `docs/assets/bench/<run-id>/` (docs/benchmarks.md:174), which is
    tracked, so the claim is checkable and is checked here.
    """
    doc = (ROOT / "docs" / "benchmarks.md").read_text(encoding="utf-8")
    assets = ROOT / "docs" / "assets" / "bench"
    assert assets.is_dir(), f"published benchmark assets are missing: {assets}"
    cited = set(re.findall(r"(20\d{6}_\d{6})", doc))
    assert cited, "no run IDs found in docs/benchmarks.md; the check is vacuous"
    published = [p.name for p in assets.iterdir() if p.is_dir()]

    missing = sorted(i for i in cited if not any(i in name for name in published))
    assert not missing, (
        f"docs/benchmarks.md cites run IDs with nothing published under "
        f"docs/assets/bench/: {missing}"
    )
    orphan = sorted(
        name for name in published if not any(name.endswith(i) for i in cited)
    )
    assert not orphan, (
        f"docs/assets/bench/ holds run data the benchmarks page never cites: "
        f"{orphan} -- either cite them or drop the directories"
    )


def test_bench_report_workflow_publishes_the_csvs_it_cites() -> None:
    """The automated benchmark PR must not be born red.

    `bench-report.yml` runs bench-all under a `ci_<timestamp>` run id, feeds it
    to `scripts/patch_bench_docs.py`, which writes that run id into the
    benchmarks page, then opens a PR that stages `docs/benchmarks.md` only --
    the CSVs go to a workflow artifact, which expires. So the page cites a run
    with nothing published, and `test_benchmark_run_ids_match_published_assets`
    (added in the .cursor/ review) fails on the very PR the workflow exists to
    produce, in the `docs-contract` job that runs on every PR. The workflow can
    only ever open an unmergeable PR.

    The fix is the documented policy rather than a test relaxation: commit the
    CSVs under `docs/assets/bench/<run-id>/`, which is what the page already
    says happens ("published with each release and mirrored under
    docs/assets/bench/<run-id>/") and what the seven tracked run directories are.
    """
    workflow = (ROOT / ".github" / "workflows" / "bench-report.yml").read_text(
        encoding="utf-8"
    )
    staged = re.findall(r"git add ([^\n]+)", workflow)
    assert staged, "bench-report.yml stages nothing; the harness below is vacuous"
    publishes = any("docs/assets/bench" in entry for entry in staged)
    assert publishes, (
        "bench-report.yml writes a run id into docs/benchmarks.md but stages no "
        "docs/assets/bench/<run-id>/ CSVs for it, so the PR it opens fails "
        f"test_benchmark_run_ids_match_published_assets. Staged: {staged}"
    )


def _cited_bench_runs(doc: str) -> list[str]:
    """Run IDs the benchmarks page's PROVENANCE comment claims to summarise."""
    match = re.search(r"<!--\s*PROVENANCE:(.*?)-->", doc, re.S)
    assert match, "docs/benchmarks.md lost its PROVENANCE comment"
    return sorted(set(re.findall(r"exhaustive_\w+", match.group(1))))


def _headline(doc: str) -> str:
    """The opening blockquote -- the hand-written summary claim block."""
    quoted = [
        line[2:].strip() if line.startswith("> ") else line[1:].strip()
        for line in doc.splitlines()
        if line.startswith(">")
    ]
    assert quoted, "docs/benchmarks.md has no opening blockquote headline"
    return " ".join(quoted)


def test_benchmarks_headline_claims_match_the_cited_runs() -> None:
    """The headline blockquote is prose, so nothing regenerated it. It rotted.

    `scripts/patch_bench_docs.py` rewrites seven marked regions of
    `docs/benchmarks.md` (`BENCH_IOPATH`, `BENCH_HIGHLIGHTS`, `BENCH_FULL_TABLE`,
    `BENCH_DEFICITS`, `BENCH_HOSTS`, `BENCH_QUICK`, `BENCH_ML`). The opening
    blockquote is not one of them: it is hand-written prose, and every run that
    refreshes the numbers below it leaves it untouched. Measured against the two
    runs its own PROVENANCE comment cites:

    1. "The remaining **significant** case family where a peer is ahead is
       **narrow-table full reads**" -- but `exhaustive_cpu_20260807_013736` has
       *three* significant deficits in *two* case families. The unmentioned one
       is `fitstable / specialized / ascii_10000 [predicate_filter] / mmap=off`
       at 1.064x, 6.40% behind, flagged `significance=significant` by
       `compute_deficits` (fitstable floor is 1.05x, so 1.064 clears it). A
       reader auditing "did the 1.2 single-pass fix land?" would check the
       narrow-table rows, see them unchanged, and conclude yes.

    2. "Image HCOMPRESS lags vs fitsio are sub-1.03x noise" -- true on CPU
       (max hcompress lag 1.0219) and **false on CUDA**, where
       `compressed_hcompress_1 [read_full @ cuda]` reaches 1.0367x (3.67% behind
       vs `fitsio_torch_device`, mmap=off). The sentence names no host while the
       blockquote explicitly covers both cited runs.

    The other two clauses check out and are pinned here so they cannot drift
    either: zero significant `fits` deficits in both runs ("100% of significant
    image comparisons"), and smart-family table win rates of 98.9% (CPU) and
    99.5% (CUDA).
    """
    doc = (ROOT / "docs" / "benchmarks.md").read_text(encoding="utf-8")
    runs = _cited_bench_runs(doc)
    assert runs, "the PROVENANCE comment names no runs; this check is vacuous"
    headline = _headline(doc)

    deficits: dict[str, list[dict[str, str]]] = {}
    for run in runs:
        path = ROOT / "docs" / "assets" / "bench" / run / "torchfits_deficits.csv"
        assert path.is_file(), f"PROVENANCE cites {run} but {path} is missing"
        with path.open(encoding="utf-8", newline="") as handle:
            deficits[run] = list(csv.DictReader(handle))
    assert any(deficits.values()), "cited runs contain no deficit rows; vacuous"

    def sig(run: str) -> list[dict[str, str]]:
        return [r for r in deficits[run] if r["significance"] == "significant"]

    # (1) A 100% image claim is only legal when no cited run has a
    # significant fits deficit. A page that names those deficits must not
    # make the claim.
    image_deficits = {
        run: [r for r in sig(run) if r["domain"] == "fits"] for run in runs
    }
    if "100% of significant image comparisons" in headline:
        assert not any(image_deficits.values()), (
            f"the headline claims 100% of significant image comparisons, but "
            f"significant fits deficits exist: {image_deficits}"
        )

    # (2) every operation that a peer wins significantly must be named.
    # Aliases keep the prose readable ("full reads" for read_full).
    aliases = {"read_full": ("read_full", "full read")}
    unnamed: set[str] = set()
    for run in runs:
        for row in sig(run):
            for form in aliases.get(row["operation"], (row["operation"],)):
                if form in headline:
                    break
            else:
                unnamed.add(f"{run}:{row['operation']} ({row['case_label']})")
    assert not unnamed, (
        "the headline calls the remaining significant case families singular but "
        "the cited runs contain significant deficits in operations it never "
        f"names: {sorted(unnamed)}"
    )

    # (3) A lag bound is attributed to the *nearest* host name in its sentence,
    # and a sentence with no host name is a claim about every cited run. So
    # "sub-1.03x noise on CPU and 1.037x on CUDA" is checkable, and an
    # unqualified "sub-1.03x" has to hold on both -- which is what the CUDA run
    # (max hcompress lag 1.0367) violates today.
    hcompress_max = {
        run: max(
            (
                float(r["lag_ratio"])
                for r in deficits[run]
                if "hcompress" in r["case_label"].lower()
            ),
            default=0.0,
        )
        for run in runs
    }
    host_runs: dict[str, list[str]] = {"CPU": [], "CUDA": [], "MPS": [], "APPLE": []}
    for run in runs:
        for host, slug in (
            ("CPU", "cpu"),
            ("CUDA", "cuda"),
            ("MPS", "mps"),
            ("APPLE", "apple"),
        ):
            if slug in run.lower():
                host_runs[host].append(run)
    assert any(host_runs.values()), (
        f"no cited run id identifies a host, so the lag bounds cannot be "
        f"attributed: {runs}"
    )

    host_re = re.compile(r"\b(CPU|CUDA|MPS|Apple)\b")
    for sentence in re.split(r"(?<=\.)\s+", headline):
        hosts = [(m.start(), m.group(1).upper()) for m in host_re.finditer(sentence)]
        for match in re.finditer(r"(\d+\.\d+)×", sentence):
            bound = float(match.group(1))
            nearest = (
                min(hosts, key=lambda h: abs(h[0] - match.start()))[1] if hosts else ""
            )
            # `Apple Silicon` is the MPS host; the prose may not repeat the name.
            if nearest == "APPLE":
                nearest = "MPS"
            applicable = host_runs[nearest] or runs
            breached = {
                r: hcompress_max[r] for r in applicable if hcompress_max[r] >= bound
            }
            assert not breached, (
                f"headline claims hcompress lags stay under {bound}x"
                f"{f' on {nearest}' if nearest else ' on every cited run'}, but the "
                f"published deficits say otherwise: {breached} (max hcompress lag "
                f"per cited run: {hcompress_max})"
            )


# ──────────────────────────────────────────────────────────────────────────
# Row 18 (docs unit). Every guard below was red before its fix; the
# measurement that made each one a finding is recorded in
# .cursor/reviews/deep-review/20-docs.md.
# ──────────────────────────────────────────────────────────────────────────
