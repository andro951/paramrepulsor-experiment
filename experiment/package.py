#!/usr/bin/env python3
"""Build and verify the runnable deliverable, excluding environments/private history."""
import hashlib,json,pathlib,subprocess,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'ParamRepulsor_Experiment.zip'
files=[ROOT/'README.md']
# Use the exact tracked source sets; never include .git credentials/cache state.
source_names=list(json.loads((ROOT/'experiment/provenance.json').read_text())['baseline_tracked_file_sha256'])
for variant in ('upstream','modified'):
 files.extend(ROOT/variant/name for name in source_names)
files.extend(p for p in (ROOT/'inputs').rglob('*') if p.is_file())
files.extend(p for p in (ROOT/'experiment').rglob('*') if p.is_file() and not any(x in p.parts for x in ('__pycache__','.pytest_cache')) and p.suffix not in ('.pyc',) and p.name not in ('paper.pdf','paper.txt','report_preview.log'))
files=sorted(set(files));assert all(p.is_file() for p in files)
checks={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files:z.write(p,'ParamRepulsor_Experiment/'+str(p.relative_to(ROOT)))
 z.writestr('ParamRepulsor_Experiment/SHA256SUMS.json',json.dumps(checks,indent=2))
with zipfile.ZipFile(OUT) as z:
 assert z.testzip() is None
 for name,h in checks.items():assert hashlib.sha256(z.read('ParamRepulsor_Experiment/'+name)).hexdigest()==h
summary={'zip':str(OUT),'bytes':OUT.stat().st_size,'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),'member_count':len(files)+1,'all_member_checksums_verified':True}
(ROOT/'package_verification.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
