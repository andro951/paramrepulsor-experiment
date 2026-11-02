"""Verify the published snapshot bytes without training or network access."""
import hashlib
import json
from pathlib import Path

def verify(root=None):
    root=Path(root) if root is not None else Path(__file__).resolve().parent
    hashes=json.loads((root/'PROJECT-CHECKSUMS.json').read_text(encoding='utf-8'))
    errors=[]
    for relative,expected in hashes.items():
        path=root/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            errors.append(relative)
    if errors:
        raise RuntimeError('Snapshot files changed or missing: '+', '.join(errors))
    return len(hashes)

if __name__=='__main__':
    print(f'PASS: {verify()} published file checksums match; no training was run.')
