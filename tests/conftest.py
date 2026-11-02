from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# The harness package lives in src/. Torchfits is the installed wheel, not this
# tree, and pixi does not install it: `pixi run subject <wheel>` does.
try:
    import torchfits  # noqa: E402
    from torchfits.cache import CACHE_ENV_SENTINELS  # noqa: E402
except ModuleNotFoundError as exc:
    if exc.name != "torchfits":
        raise
    torchfits = None
    CACHE_ENV_SENTINELS = ()

NO_SUBJECT = (
    "torchfits not installed: run `pixi run subject <wheel-or-spec>` "
    "(or `pixi run subject-local` to build ../torchfits)"
)


def _subject_line() -> str:
    if torchfits is None:
        return "torchfits subject: NOT INSTALLED"
    return f"torchfits subject: {torchfits.__version__} {torchfits.__file__}"


def pytest_report_header(config):
    return _subject_line()


def pytest_collectreport(report):
    # A module that imports torchfits at top level fails collection; stop there
    # with the instruction instead of a wall of ImportErrors.
    if (
        torchfits is None
        and report.failed
        and "No module named 'torchfits'" in str(report.longrepr)
    ):
        pytest.exit(NO_SUBJECT, returncode=4)


def pytest_collection_modifyitems(config, items):
    # Only tests marked `independent` (layout, docs, manifests) run without it.
    if torchfits is None and any(
        item.get_closest_marker("independent") is None for item in items
    ):
        pytest.exit(NO_SUBJECT, returncode=4)


# Several CACHE_ENV_SENTINELS (KUBERNETES_SERVICE_HOST, K_SERVICE, WEBSITE_SITE_NAME, ...)
# leak into dev sandboxes and shared CI pods, where torchfits environment detection would
# otherwise mis-classify the env as "cloud" / "hpc". The autouse fixture below clears the
# FULL allowlist (single source of truth: CACHE_ENV_SENTINELS imported above from
# torchfits.cache) before every test so each test starts from a clean env and can opt
# into its expected profile by setenv'ing the relevant sentinel. Tests that explicitly
# opt in (e.g. monkeypatch.setenv SLURM_JOB_ID for hpc tests, AWS_EXECUTION_ENV for cloud
# tests) still work because monkeypatch.setenv runs AFTER the fixture's delenv pass.
@pytest.fixture(autouse=True)
def _clear_env_profile(monkeypatch):
    for var in CACHE_ENV_SENTINELS:
        monkeypatch.delenv(var, raising=False)
    yield


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Name the subject and say how many tests skipped for an unfetched corpus.

    The header is suppressed under ``-q``, which every task uses, so the subject
    is repeated here. The corpus is gitignored, so on a fresh clone those skips
    are expected; the line keeps them from passing for real coverage.
    """
    if config.get_verbosity() < 0:
        terminalreporter.write_line(_subject_line())
    skipped = terminalreporter.stats.get("skipped", [])
    missing = [
        report
        for report in skipped
        if "is not present; run `pixi run corpus-fetch" in str(report.longrepr)
    ]
    if missing:
        terminalreporter.write_line(
            f"{len(missing)} test(s) skipped: corpus files not fetched "
            "(`pixi run corpus-fetch` for the CI tier, "
            "`pixi run corpus-fetch-full` for everything)"
        )
