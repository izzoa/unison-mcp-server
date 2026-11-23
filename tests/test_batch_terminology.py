"""One meaning of "batch" (add-batch-model-support, section 12).

Before provider batch execution existed, consensus called its concurrent step-1 dispatch (every model
consulted at once with ``asyncio.gather``) "batch dispatch". "Batch" now means one thing: a provider's
asynchronous Batch API. These guards fail if the old meaning comes back in the consensus tool, the tests, or
the docs, so that a reader grepping for "batch" finds only provider-batch behaviour.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Assembled from parts so this file does not match itself.
OLD_MEANING = re.compile("batch" + r"[\s_-]*" + "dispatch(?:ed|ing)?", re.IGNORECASE)
# "batch" used for the concurrent fan-out: "batch ... concurrently", "concurrent batch of consultations".
CONCURRENCY_SENSE = re.compile(r"batch[^\n]{0,40}concurren|concurren[^\n]{0,40}batch", re.IGNORECASE)


def _scan(paths, pattern):
    hits = []
    for path in paths:
        if path == Path(__file__).resolve():
            continue  # this file states the old phrase in order to forbid it
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    return hits


def _python_and_docs() -> list[Path]:
    roots = [ROOT / "tools", ROOT / "tests", ROOT / "docs", ROOT / "utils", ROOT / "simulator_tests"]
    files = [p for r in roots for p in r.rglob("*") if p.suffix in {".py", ".md"} and "__pycache__" not in p.parts]
    files += [ROOT / "README.md", ROOT / "CHANGELOG.md"]
    return files


class TestConsensusSource:
    def test_the_step_one_dispatch_comment_describes_concurrency_without_the_word_batch(self):
        source = (ROOT / "tools" / "consensus.py").read_text(encoding="utf-8")

        assert "# Dispatch all model consultations concurrently on step 1" in source
        assert not OLD_MEANING.search(source)

    def test_every_batch_mention_in_the_consensus_tool_is_about_a_provider_batch(self):
        """No line pairs "batch" with the concurrent fan-out unless it is contrasting the two."""

        path = ROOT / "tools" / "consensus.py"
        offenders = [h for h in _scan([path], CONCURRENCY_SENSE) if "run concurrently as they" not in h]

        assert offenders == []


class TestWholeRepository:
    def test_no_file_uses_the_old_name_for_concurrent_dispatch(self):
        assert _scan(_python_and_docs(), OLD_MEANING) == []

    def test_tests_pair_batch_with_concurrency_only_when_both_are_about_provider_batches(self):
        """12.2: the audited exceptions are provider-batch races (two reconcilers, a cancel racing a resume)."""

        hits = _scan(sorted((ROOT / "tests").rglob("*.py")), CONCURRENCY_SENSE)
        allowed = ("test_batch_coordinator.py", "test_batch_execution.py")

        assert [h for h in hits if not h.startswith(tuple(f"tests/{name}" for name in allowed))] == []

    def test_the_guard_pattern_catches_the_old_phrasing(self):
        for text in ("# Batch-dispatch all model consultations", "batch dispatch on first step", "BATCH_DISPATCH"):
            assert OLD_MEANING.search(text), text
        assert not OLD_MEANING.search("# Dispatch all model consultations concurrently on step 1")
        assert not OLD_MEANING.search("one provider batch per distinct model")

    def test_the_concurrency_pattern_catches_batch_used_for_the_fan_out(self):
        assert CONCURRENCY_SENSE.search("# send the batch of consultations concurrently")
        assert CONCURRENCY_SENSE.search("# concurrently dispatch the whole batch")
        assert CONCURRENCY_SENSE.search("# step 1 is a batch: every model at once, concurrently")
