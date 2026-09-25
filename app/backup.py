"""Container-runnable, transaction-consistent SQLite backup.

Usage inside the running container::

    python -m app.backup --output /data/backups/ledger-20260925.db

The snapshot is taken with SQLite's *online backup API*
(``sqlite3.Connection.backup``), not by copying the database file.  The API
runs concurrently with the ledger service: pages are copied in small batches
and a writer commit mid-copy forces a restart from the beginning instead of
exposing half of a transaction, so the resulting file is always one coherent
snapshot -- every committed request, reservation, batch, transfer or reversal
is present in full or not at all.

Safety guarantees:

* the destination is never overwritten -- an existing ``--output`` path (or a
  file published by a concurrent backup) makes the command fail with a clear
  message and non-zero exit code;
* the snapshot is first written to a temporary file in the same directory and
  passes ``PRAGMA integrity_check`` (plus a foreign-key check) before it is
  published;
* publication is atomic (hard link into place + directory fsync); any failure
  leaves no temporary file and no partial destination;
* the source database is never modified.

The default output directory in the container should be under ``/data`` (the
persistent named volume) so backups survive container rebuilds.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

from .config import settings

# Pages copied per backup-API batch.  With the service's single fast writer
# small batches are plenty; whenever the source commits between batches the
# backup transparently restarts, so a torn transaction can never be copied.
PAGES_PER_BATCH = 100
# Delay between batches; also the sleep the backup API uses while retrying a
# locked source database.
RETRY_SLEEP_SECONDS = 0.05


class BackupError(Exception):
    """The backup could not be produced; ``str(exc)`` is the clear reason."""


def create_backup(db_path: str, output_path: str) -> str:
    """Create a consistent backup of *db_path* at *output_path*.

    Returns the published output path.  Raises :class:`BackupError` with a
    human-readable reason on any failure; on failure the temporary file is
    removed and an existing destination is never overwritten.
    """
    source = Path(db_path)
    destination = Path(output_path)

    if not source.is_file():
        raise BackupError(f"source database not found: {db_path}")
    if source.stat().st_size == 0:
        raise BackupError(
            f"source database is empty (not an initialised SQLite database): "
            f"{db_path}"
        )
    try:
        with open(db_path, "rb") as header_file:
            header = header_file.read(16)
    except OSError as exc:
        raise BackupError(f"cannot read source database: {exc}") from exc
    if header != b"SQLite format 3\x00":
        raise BackupError(
            f"source file is not a SQLite database (bad magic header): "
            f"{db_path}"
        )
    with contextlib.suppress(OSError):
        if source.resolve() == destination.resolve():
            raise BackupError(
                "output path must not be the live database file itself: "
                f"{output_path}"
            )
    if destination.exists():
        raise BackupError(
            "backup destination already exists, refusing to overwrite: "
            f"{output_path}"
        )

    parent = destination.parent
    if parent.exists() and not parent.is_dir():
        raise BackupError(
            f"output path's parent is not a directory: {parent}"
        )
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BackupError(
            f"cannot create output directory {parent}: {exc}"
        ) from exc

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".backup-tmp", dir=str(parent)
    )
    os.close(fd)
    temporary = Path(tmp_name)

    published = False
    try:
        _online_copy(db_path, tmp_name)
        _verify_backup(tmp_name)
        # mkstemp defaults to 0600; match a normal SQLite data file so the
        # backup can be restored with the same ownership/permissions workflow.
        os.chmod(tmp_name, 0o644)
        _fsync_file(tmp_name)
        # Atomic, non-clobbering publish: os.link fails if the destination
        # appeared between the earlier check and now (e.g. another backup
        # racing to the same path), and rename is deliberately not used so an
        # existing backup can never be replaced.
        try:
            os.link(tmp_name, destination)
        except FileExistsError:
            raise BackupError(
                "backup destination already exists, refusing to overwrite: "
                f"{output_path}"
            ) from None
        except OSError as exc:
            raise BackupError(
                f"cannot publish backup to {output_path}: {exc}"
            ) from exc
        published = True
        _fsync_directory(parent)
    except BaseException:
        if not published:
            with contextlib.suppress(OSError):
                temporary.unlink()
        raise
    else:
        # The file now lives at the destination; drop the extra directory
        # entry for the temporary file (best effort, same filesystem).
        with contextlib.suppress(OSError):
            temporary.unlink()

    return str(destination)


def _online_copy(db_path: str, tmp_path: str) -> None:
    """Copy a transaction-consistent snapshot via the online backup API."""
    # A separate process-local, strictly read-only connection (mode=ro means
    # the backup can never modify the live database): in WAL mode it reads
    # alongside the running service without blocking or being blocked by its
    # writers, and busy_timeout makes it queue behind a writer instead of
    # failing outright.
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    source = sqlite3.connect(
        uri,
        uri=True,
        timeout=max(1.0, settings.busy_timeout_ms / 1000.0),
    )
    target = sqlite3.connect(tmp_path)
    try:
        source.execute(f"PRAGMA busy_timeout = {int(settings.busy_timeout_ms)}")
        try:
            source.backup(
                target,
                pages=PAGES_PER_BATCH,
                sleep=RETRY_SLEEP_SECONDS,
            )
        except sqlite3.Error as exc:
            raise BackupError(f"SQLite online backup failed: {exc}") from exc
        # The backup inherits the source's WAL file header.  Switch the fresh
        # artifact to the rollback journal while it is still the only
        # connection so it is fully self-contained (no -wal/-shm sidecars);
        # the service switches it back to WAL on startup.
        try:
            mode = target.execute("PRAGMA journal_mode=DELETE").fetchone()
        except sqlite3.Error as exc:
            raise BackupError(
                f"could not finalize backup journal mode: {exc}"
            ) from exc
        if mode is None or str(mode[0]).lower() != "delete":
            raise BackupError(
                "could not finalize backup as a self-contained database file "
                f"(journal_mode={mode[0] if mode else 'unknown'})"
            )
    finally:
        target.close()
        source.close()


def _verify_backup(path: str) -> None:
    """Reject the artifact unless SQLite itself vouches for its integrity."""
    conn = sqlite3.connect(path)
    try:
        try:
            row = conn.execute("PRAGMA integrity_check").fetchone()
        except sqlite3.Error as exc:
            raise BackupError(f"integrity_check failed to run: {exc}") from exc
        if row is None or row[0] != "ok":
            detail = row[0] if row else "no result"
            raise BackupError(f"PRAGMA integrity_check failed: {detail}")
        try:
            violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        except sqlite3.Error as exc:
            raise BackupError(f"foreign_key_check failed to run: {exc}") from exc
        if violations:
            raise BackupError(
                f"foreign_key_check reported {len(violations)} "
                "violating row(s)"
            )
    finally:
        conn.close()


def _fsync_file(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError as exc:
        raise BackupError(f"cannot flush backup file {path}: {exc}") from exc
    finally:
        os.close(fd)


def _fsync_directory(directory: Path) -> None:
    """Persist the new directory entry (the hard-link publish) to disk."""
    fd = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError as exc:
        raise BackupError(
            f"cannot flush output directory {directory}: {exc}"
        ) from exc
    finally:
        os.close(fd)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.backup",
        description=(
            "Take a transaction-consistent online backup of the ledger's "
            "SQLite database (DATABASE_PATH) and atomically publish it to "
            "--output. The destination is never overwritten."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help=(
            "backup file path to create; must not already exist. Inside the "
            "container keep it under /data so it survives rebuilds, e.g. "
            "/data/backups/ledger-YYYYMMDD.db"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        published = create_backup(settings.db_path, args.output)
    except BackupError as exc:
        print(f"backup failed: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"backup failed: {exc}", file=sys.stderr)
        return 1
    print(f"backup of {settings.db_path} written to {published}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
