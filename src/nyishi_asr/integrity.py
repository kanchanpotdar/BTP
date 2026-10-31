"""Hashing, safe copies and complete artifact publication."""
from pathlib import Path
import hashlib
import os
import shutil
import tempfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def copy_exact(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if digest(source) != digest(destination):
            raise RuntimeError(f'Refusing to overwrite different artifact: {destination}')
        return
    fd, temporary = tempfile.mkstemp(prefix='.' + destination.name, dir=destination.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        if digest(source) != digest(temporary):
            raise RuntimeError(f'Copy hash mismatch: {source}')
        # Hard link is an atomic no-clobber publication on this same filesystem.
        os.link(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def publish(stage_root, project_root):
    files = [p for folder in ['manifests', 'cleaning', 'metadata']
             for p in (Path(stage_root) / folder).rglob('*') if p.is_file()]
    # The current recovery log is append-only across validated stages.
    # All other published artifacts must be byte-identical or absent.
    log_updates = []
    for source in files:
        dest = Path(project_root) / source.relative_to(stage_root)
        if dest.exists() and digest(dest) != digest(source):
            if str(source.relative_to(stage_root)) == 'cleaning/decision_log_A.csv' and source.read_bytes().startswith(dest.read_bytes()):
                log_updates.append((source, dest, digest(dest)))
            else:
                raise RuntimeError(f'Publication conflict, original retained: {dest}')
    updates = {src for src, _, _ in log_updates}
    for source in files:
        if source not in updates:
            copy_exact(source, Path(project_root) / source.relative_to(stage_root))
    for source, dest, previous_hash in log_updates:
        if digest(dest) != previous_hash:
            raise RuntimeError('Recovery log changed during publication')
        fd, temporary = tempfile.mkstemp(prefix='.decision_log_', dir=dest.parent)
        os.close(fd)
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, dest)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return [str(p.relative_to(stage_root)) for p in files]
