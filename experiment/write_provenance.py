import hashlib,json,pathlib,platform,subprocess,os
ROOT=pathlib.Path(__file__).resolve().parents[1]
previous=json.loads((ROOT/'experiment/provenance.json').read_text()) if (ROOT/'experiment/provenance.json').exists() else {}
has_git=(ROOT/'upstream/.git').exists()
files=subprocess.check_output(['git','-C',str(ROOT/'upstream'),'ls-files','-z']).decode().split('\0') if has_git else list(previous['baseline_tracked_file_sha256']);files=[x for x in files if x]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
baseline={f:sha(ROOT/'upstream'/f) for f in files};modified={f:sha(ROOT/'modified'/f) for f in files};changed=[f for f in files if baseline[f]!=modified[f]]
assert changed==['src/parampacmap/utils/data.py'],changed
if has_git:assert not subprocess.check_output(['git','-C',str(ROOT/'upstream'),'status','--porcelain']).strip()
else:assert baseline==previous['baseline_tracked_file_sha256']
inputs={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'inputs').rglob('*') if p.is_file()}
provenance={'baseline_repository':'https://github.com/hyhuang00/ParamRepulsor','baseline_commit':(subprocess.check_output(['git','-C',str(ROOT/'upstream'),'rev-parse','HEAD']).decode().strip() if has_git else previous['baseline_commit']),'paper':'https://arxiv.org/abs/2411.15894','paper_pdf_sha256':(sha(ROOT/'experiment/paper.pdf') if (ROOT/'experiment/paper.pdf').exists() else previous['paper_pdf_sha256']),'input_library_file_id':'libfile_59b61f1348648191b85bb99fdd49d9cb','input_sha256':inputs,'baseline_tracked_file_sha256':baseline,'modified_tracked_file_sha256':modified,'changed_source_files':changed,'python':platform.python_version(),'platform':platform.platform(),'cpu_count_visible':os.cpu_count(),'quality_trials':{'datasets':[36,50,100],'seeds':list(range(10)),'variants':['upstream','modified'],'total':60,'illustrative_seed':0,'outcome_selection':'all outcomes retained; illustrative seed0 fixed before outcome inspection'},'runtime_adapter':'experiment/run_trial.py: inference-only, single original batch in-process because sandbox denies PyTorch worker IPC; identical arms','publication':'none; no upstream push or public deployment'}
(ROOT/'experiment/provenance.json').write_text(json.dumps(provenance,indent=2));print(json.dumps({k:provenance[k] for k in ['baseline_commit','changed_source_files','python','cpu_count_visible']},indent=2))
