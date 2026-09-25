"""Tests for the container-runnable online backup (``python -m app.backup``).

Covers the consistency contract: a snapshot taken via the SQLite online
backup API while a writer commits must contain only whole transactions,
pass integrity checks, publish atomically and never overwrite an existing
backup.
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from app.backup import BackupError, create_backup

REPO_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ helpers --


def _init_source(path: Path) -> sqlite3.Connection:
    """Create a WAL database resembling a migrated ledger."""
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(
        """
        CREATE TABLE datasets (
            dataset_id TEXT PRIMARY KEY, epsilon_cap REAL NOT NULL
        );
        CREATE TABLE requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
            epsilon REAL NOT NULL
        );
        CREATE TABLE transfer_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, total REAL NOT NULL
        );
        INSERT INTO datasets VALUES ('sales-2026', 1.0);
        """
    )
    conn.commit()
    return conn


@pytest.fixture()
def workdir(tmp_path: Path) -> Path:
    return tmp_path


# -------------------------------------------------------------------- happy --


def test_backup_contains_latest_committed_snapshot(workdir):
    source = workdir / "ledger.db"
    conn = _init_source(source)
    conn.execute(
        "INSERT INTO requests (dataset_id, epsilon) VALUES ('sales-2026', 0.2)"
    )
    conn.commit()
    conn.close()

    out = workdir / "backups" / "ledger-20260925.db"
    published = create_backup(str(source), str(out))

    assert Path(published) == out
    assert out.is_file()
    assert out.stat().st_mode & 0o777 == 0o644
    # No temporary leftovers in the output directory.
    assert not list(out.parent.glob("*.backup-tmp"))

    restored = sqlite3.connect(str(out))
    try:
        assert restored.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        # The artifact is a self-contained rollback-journal database: no
        # -wal/-shm sidecars are needed to restore it.
        assert str(restored.execute("PRAGMA journal_mode").fetchone()[0]) == (
            "delete"
        )
        rows = restored.execute(
            "SELECT dataset_id, epsilon FROM requests ORDER BY id"
        ).fetchall()
        assert rows == [("sales-2026", 0.2)]
    finally:
        restored.close()
    assert not Path(str(out) + "-wal").exists()
    assert not Path(str(out) + "-shm").exists()


def test_backup_creates_missing_parent_directories(workdir):
    source = workdir / "ledger.db"
    conn = _init_source(source)
    conn.close()
    out = workdir / "nested" / "deep" / "backup.db"

    create_backup(str(source), str(out))

    assert out.is_file()


# -------------------------------------------------------------- error paths --


def test_existing_destination_is_never_overwritten(workdir):
    source = workdir / "ledger.db"
    conn = _init_source(source)
    conn.close()
    out = workdir / "backup.db"
    out.write_bytes(b"PRE-EXISTING BACKUP")

    with pytest.raises(BackupError, match="already exists"):
        create_backup(str(source), str(out))

    assert out.read_bytes() == b"PRE-EXISTING BACKUP"
    assert not list(workdir.glob("*.backup-tmp"))


def test_missing_source_reports_clear_reason(workdir):
    out = workdir / "backup.db"
    with pytest.raises(BackupError, match="source database not found"):
        create_backup(str(workdir / "missing.db"), str(out))
    assert not out.exists()
    assert not list(workdir.glob("*.backup-tmp"))


def test_output_equal_to_source_rejected(workdir):
    source = workdir / "ledger.db"
    conn = _init_source(source)
    conn.close()
    with pytest.raises(BackupError, match="must not be the live database"):
        create_backup(str(source), str(source.resolve()))


def test_non_sqlite_source_rejected_and_no_artifact(workdir):
    source = workdir / "ledger.db"
    source.write_bytes(b"this is not a database" * 4)
    out = workdir / "backup.db"
    with pytest.raises(BackupError, match="not a SQLite database"):
        create_backup(str(source), str(out))
    assert not out.exists()
    assert not list(workdir.glob("*.backup-tmp"))


def test_corrupt_backup_is_discarded_not_published(workdir, monkeypatch):
    source = workdir / "ledger.db"
    conn = _init_source(source)
    conn.close()
    out = workdir / "backup.db"

    import app.backup as backup_mod

    def broken_verify(path):
        # Truncate the artifact after the copy, simulating an I/O fault that
        # leaves an incomplete database; integrity checking must reject it.
        size = os.path.getsize(path)
        with open(path, "r+b") as fh:
            fh.truncate(max(0, size // 2))
        raise BackupError("PRAGMA integrity_check failed: simulated fault")

    monkeypatch.setattr(backup_mod, "_verify_backup", broken_verify)
    with pytest.raises(BackupError, match="simulated fault"):
        create_backup(str(source), str(out))

    assert not out.exists()
    assert not list(workdir.glob("*.backup-tmp"))


# ----------------------------------------------------- concurrency / atomicity --


def test_snapshot_consistent_under_concurrent_writer(workdir):
    """A backup taken mid-writes must never expose a half transaction.

    The writer applies two-row transactions with an invariant the ledger
    relies on (request count == transfer count == a shared counter); a torn
    transaction would show mismatched counts.
    """
    source = workdir / "ledger.db"
    conn = _init_source(source)
    conn.close()

    stop = threading.Event()
    writer_errors: list[Exception] = []
    committed = [0]

    def writer():
        wc = sqlite3.connect(str(source), timeout=30)
        try:
            while not stop.is_set():
                n = committed[0] + 1
                with wc:
                    wc.execute("BEGIN IMMEDIATE")
                    wc.execute(
                        "INSERT INTO requests (dataset_id, epsilon) "
                        "VALUES ('sales-2026', ?)",
                        (0.001 * n,),
                    )
                    wc.execute(
                        "INSERT INTO transfer_log (total) VALUES (?)", (n,)
                    )
                committed[0] = n
                time.sleep(0)  # yield to the backup
        except Exception as exc:  # pragma: no cover - failure reporter
            writer_errors.append(exc)
        finally:
            wc.close()

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        # Wait until the writer has actually committed.
        deadline = time.monotonic() + 5
        while committed[0] == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert committed[0] > 0

        backups = []
        for i in range(3):
            out = workdir / f"backup-{i}.db"
            create_backup(str(source), str(out))
            backups.append(out)
    finally:
        stop.set()
        thread.join(timeout=10)

    assert writer_errors == []

    for out in backups:
        bc = sqlite3.connect(str(out))
        try:
            assert bc.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            assert bc.execute("PRAGMA foreign_key_check").fetchall() == []
            request_count = bc.execute(
                "SELECT COUNT(*) FROM requests"
            ).fetchone()[0]
            transfer_count = bc.execute(
                "SELECT COUNT(*) FROM transfer_log"
            ).fetchone()[0]
            # Whole-transaction consistency: the two tables are written in
            # one commit; their counts must always agree and every transfer
            # row must be paired with a request row at the same sequence.
            assert request_count == transfer_count
            pairs = bc.execute(
                """
                SELECT r.idx_eps, t.total FROM (
                    SELECT epsilon AS idx_eps,
                           ROW_NUMBER() OVER (ORDER BY id) AS rn
                    FROM requests
                ) r JOIN (
                    SELECT total, ROW_NUMBER() OVER (ORDER BY id) AS rn
                    FROM transfer_log
                ) t ON r.rn = t.rn
                """
            ).fetchall()
            for idx_eps, total in pairs:
                assert idx_eps == pytest.approx(0.001 * total)
        finally:
            bc.close()


def test_concurrent_backups_to_same_path_publish_exactly_once(workdir):
    """Two backups racing to one destination must never clobber each other."""
    source = workdir / "ledger.db"
    conn = _init_source(source)
    conn.close()
    out = workdir / "race.db"

    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def worker():
        barrier.wait()
        try:
            create_backup(str(source), str(out))
            outcomes.append("published")
        except BackupError:
            outcomes.append("rejected")

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert sorted(outcomes) == ["published", "rejected"]
    assert out.is_file()
    # No temporary files leaked.
    assert not list(workdir.glob("*.backup-tmp"))
    restored = sqlite3.connect(str(out))
    try:
        assert restored.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    finally:
        restored.close()


# --------------------------------------------------------------- restore ---


def test_backup_restores_to_a_working_database(workdir):
    source = workdir / "ledger.db"
    conn = _init_source(source)
    for i in range(5):
        conn.execute(
            "INSERT INTO requests (dataset_id, epsilon) VALUES "
            "('sales-2026', ?)",
            (0.01 * (i + 1),),
        )
    conn.commit()
    conn.close()

    backup_path = workdir / "backups" / "ledger.db"
    create_backup(str(source), str(backup_path))

    restored = workdir / "restored" / "ledger.db"
    restored.parent.mkdir()
    # Restore is a plain file copy; the service switches it back to WAL.
    restored.write_bytes(backup_path.read_bytes())

    rc = sqlite3.connect(str(restored))
    try:
        assert rc.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert rc.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 5
        rc.execute("PRAGMA journal_mode=WAL")
        rc.execute(
            "INSERT INTO requests (dataset_id, epsilon) VALUES "
            "('sales-2026', 0.9)"
        )
        rc.commit()
        assert rc.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 6
    finally:
        rc.close()


# --------------------------------------------------------------------- CLI --


def test_cli_success_and_failure_exit_codes(workdir):
    source = workdir / "ledger.db"
    conn = _init_source(source)
    conn.close()
    out = workdir / "backups" / "cli.db"
    env = {
        **os.environ,
        "DATABASE_PATH": str(source),
        "PYTHONPATH": str(REPO_ROOT),
    }

    proc = subprocess.run(
        [sys.executable, "-m", "app.backup", "--output", str(out)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert str(out) in proc.stdout
    assert out.is_file()

    # Repeating onto the existing backup fails non-zero with a clear reason.
    proc = subprocess.run(
        [sys.executable, "-m", "app.backup", "--output", str(out)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 1
    assert "already exists" in proc.stderr

    # Missing --output is a usage error (non-zero).
    proc = subprocess.run(
        [sys.executable, "-m", "app.backup"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode != 0
    assert "--output" in proc.stderr

    # Missing source database.
    env_missing = {**env, "DATABASE_PATH": str(workdir / "gone.db")}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.backup",
            "--output",
            str(workdir / "never.db"),
        ],
        capture_output=True,
        text=True,
        env=env_missing,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 1
    assert "source database not found" in proc.stderr
    assert not (workdir / "never.db").exists()


def test_main_returns_zero_on_success_and_one_on_error(workdir, monkeypatch):
    source = workdir / "ledger.db"
    conn = _init_source(source)
    conn.close()
    out = workdir / "main.db"

    import dataclasses

    from app import backup as backup_mod

    # Settings is frozen; swap the module-level reference for the test run.
    monkeypatch.setattr(
        backup_mod,
        "settings",
        dataclasses.replace(backup_mod.settings, db_path=str(source)),
    )

    assert backup_mod.main(["--output", str(out)]) == 0
    assert out.is_file()
    # Existing destination: non-zero exit, clear reason, no overwrite.
    assert backup_mod.main(["--output", str(out)]) == 1
