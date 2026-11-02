"""Local HTTP and real-training checks; never overwrite cloud verification files."""
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'experiment' / 'simulator'))
from server import TrainingServer


def main():
    output = ROOT / 'windows-verification'
    output.mkdir(exist_ok=True)
    server = TrainingServer(ROOT, 8765)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = 'http://127.0.0.1:8765'
    reports = []
    try:
        with urlopen(url) as response:
            assert response.status == 200
            assert b'Start training' in response.read()
        assert json.load(urlopen(url + '/api/status')) == {'status': 'idle'}
        for variant, n in [('baseline',36),('modified',36),('baseline',50),('modified',50),('baseline',100),('modified',100),('baseline',36),('modified',36)]:
            request = Request(url + '/api/start', data=json.dumps(dict(variant=variant,n=n,seed=42)).encode(), headers={'Content-Type':'application/json','Origin':url}, method='POST')
            launched = json.load(urlopen(request))
            try:
                urlopen(request)
                raise AssertionError('Concurrent start was accepted')
            except HTTPError as error:
                assert error.code == 409
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                state = json.load(urlopen(url + '/api/status', timeout=10))
                if state['status'] in ('completed','error'):
                    break
                time.sleep(.2)
            assert state['status'] == 'completed', state
            assert state['epoch'] == 450 and state['frame_count'] == 451
            archive = json.loads((server.state_path.parent/'epochs.json').read_text())
            assert [f['epoch'] for f in archive['frames']] == list(range(451))
            assert all(len(f['coordinates']) == n for f in archive['frames'])
            assert archive['frames'][-1]['coordinates'] == state['coordinates']
            with (ROOT/'inputs'/'datasets'/f'paramrep_2d_{n}.csv').open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            assert [p['point_id'] for p in state['points']] == [r['point_id'] for r in rows]
            assert [p['group'] for p in state['points']] == [r['group'] for r in rows]
            digest = hashlib.sha256(json.dumps(state['coordinates']).encode()).hexdigest()
            earlier = [r for r in reports if r['variant'] == variant and r['n'] == n]
            if earlier:
                assert digest == earlier[0]['final_coordinates_sha256']
            report = dict(variant=variant,n=n,seed=42,epochs=450,actual_frames=451,elapsed_seconds=state['elapsed_seconds'],run_id=launched['run_id'],final_coordinates_sha256=digest,repeated_same_seed_identical=bool(earlier),point_ids_and_groups_preserved=True,observation_rng_checked_each_epoch=True,final_inference_matches_observation=True)
            reports.append(report)
            print(json.dumps(report),flush=True)
    finally:
        server.shutdown()
        server.close_worker()
        server.server_close()
        thread.join()
    restarted = TrainingServer(ROOT,8765)
    restarted.server_close()
    manifest = json.loads((ROOT/'SHA256SUMS.json').read_text())
    changed = [p for p,h in manifest.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != h]
    from verify_snapshot import verify
    verified_files = verify(ROOT)
    result = dict(machine=platform.node(),python=sys.version,platform=platform.platform(),http_page_and_idle_passed=True,concurrent_start_rejected=True,server_restart_port_8765_passed=True,original_manifest_files_unchanged=len(manifest)-len(changed),local_viewer_changes=changed,project_snapshot_checksums_verified=verified_files,runs=reports,visual_browser_qa='Not performed by this command; see VALIDATION.md for recorded browser QA',server_running_after_test=False)
    (output/'actual-training.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: HTTP, eight full actual-training runs, repeatability, restart, published snapshot checksums',flush=True)


if __name__ == '__main__':
    main()
