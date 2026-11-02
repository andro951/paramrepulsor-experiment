"""Exercise real 450-epoch training through HTTP (no browser dependencies)."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['baseline', 'modified'], default='baseline')
    parser.add_argument('--port', type=int, default=18765)
    args = parser.parse_args()
    url = f'http://127.0.0.1:{args.port}'
    log = tempfile.TemporaryFile(mode='w+')
    server = subprocess.Popen([sys.executable, str(HERE/'server.py'), '--port', str(args.port)], stdout=log, stderr=log)
    report = {'variant': args.variant, 'n': 36, 'seed': 42, 'checks': []}
    try:
        for _ in range(100):
            try:
                status = json.load(urlopen(url+'/api/status', timeout=2))
                break
            except OSError:
                time.sleep(.1)
        else:
            raise RuntimeError('Local server did not start')
        assert status == {'status': 'idle'}, status
        report['checks'].append('initial idle')
        request = lambda: Request(url+'/api/start', data=json.dumps({'variant':args.variant,'n':36,'seed':42}).encode(), headers={'Content-Type':'application/json'}, method='POST')
        launch = json.load(urlopen(request()))
        report['checks'].append('Start accepted')
        try:
            urlopen(request())
            raise AssertionError('Repeated start unexpectedly accepted')
        except HTTPError as e:
            assert e.code == 409, e.code
        report['checks'].append('repeated Start rejected with 409')
        seen = []
        started = time.time()
        while time.time()-started < 300:
            state = json.load(urlopen(url+'/api/status'))
            if state.get('epoch') is not None and (not seen or seen[-1] != state['epoch']):
                seen.append(state['epoch'])
            if state['status'] in ('error','completed'):
                break
            time.sleep(.1)
        assert state['status']=='completed', state
        assert state['epoch']==450 and state['frame_count']==451, state
        assert len(state['coordinates'])==36 and len(state['initial_coordinates'])==36
        report['checks'].extend(['real training reached epoch 450','451 actual epoch frames saved','36 point identities preserved','final observed coordinates match inference','each observation preserves torch RNG'])
        report['elapsed_seconds']=state['elapsed_seconds']
        report['observed_epoch_values']=seen
        report['run_id']=launch['run_id']
        report['state']=state
        first_coordinates = state['coordinates']
        second_launch = json.load(urlopen(request()))
        assert second_launch['run_id'] != launch['run_id']
        started = time.time()
        while time.time()-started < 300:
            second = json.load(urlopen(url+'/api/status'))
            if second['status'] in ('error','completed'):
                break
            time.sleep(.1)
        assert second['status']=='completed', second
        assert second['coordinates']==first_coordinates, 'Repeated matched seed changed final output'
        report['checks'].extend(['Start after completion launches new process','repeated same-seed run completes with exactly identical final coordinates'])
        report['second_run_id'] = second_launch['run_id']
        path=HERE/f'verified_{args.variant}.json'
        path.write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k!='state'},indent=2))
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill(); server.wait()
        log.seek(0)
        print(log.read())
        log.close()

if __name__=='__main__':
    main()
