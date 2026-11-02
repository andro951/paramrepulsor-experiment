"""Local-only live ParamRepulsor viewer. Run with the experiment Python environment."""
from __future__ import annotations

import argparse
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from urllib.parse import parse_qs, urlsplit

HERE = Path(__file__).resolve().parent


class TrainingServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, root: Path, port: int):
        super().__init__(('127.0.0.1', port), Handler)
        self.root = root
        self.lock = threading.Lock()
        self.process = None
        self.run_id = None
        self.state_path = None
        self.log_handle = None
        self.run_directory = Path(tempfile.mkdtemp(prefix='paramrepulsor-live-'))

    def status(self):
        if self.process is None:
            return dict(status='idle')
        state = dict(status='preparing',replay_version=1,stage='reading_state')
        if self.state_path.exists():
            try:
                state = json.loads(self.state_path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                pass  # Atomic replace makes this exceptional; try the next poll.
        code = self.process.poll()
        if code is None and state['status'] == 'completed':
            state['status'] = 'finalizing'
        if code is not None and state['status'] not in ('completed', 'error'):
            state = dict(status='error', error=f'Training process exited with code {code}; inspect {self.log_handle.name}.')
        state['run_id'] = self.run_id
        return state

    def start(self, request):
        if set(request) != {'variant', 'n', 'seed'}:
            raise ValueError('Expected exactly variant, n and seed.')
        if request['variant'] not in ('baseline', 'modified'):
            raise ValueError('variant must be baseline or modified.')
        if type(request['n']) is not int or request['n'] not in (36, 50, 100):
            raise ValueError('n must be 36, 50 or 100.')
        if type(request['seed']) is not int or not 0 <= request['seed'] < 2**31 - 100000:
            raise ValueError('seed must be an integer from 0 through 2147383647.')
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return None
            if self.log_handle:
                self.log_handle.close()
            self.run_id = uuid.uuid4().hex
            directory = self.run_directory / self.run_id
            directory.mkdir()
            self.state_path = directory / 'state.json'
            self.state_path.write_text(json.dumps(dict(status='preparing', stage='starting_worker', replay_version=1,compute_started_at=time.time(),
                                                       variant=request['variant'],n=request['n'],seed=request['seed'],epoch=0,total_epochs=450,
                                                       points=read_points(self.root,request['n']),coordinates=None,elapsed_seconds=0,pairs_ready=False)),encoding='utf-8')
            self.log_handle = (directory / 'training.log').open('w', encoding='utf-8')
            command = [sys.executable, str(HERE / 'worker.py'), '--root', str(self.root),
                       '--variant', request['variant'], '--n', str(request['n']),
                       '--seed', str(request['seed']), '--state', str(self.state_path)]
            self.process = subprocess.Popen(command, stdout=self.log_handle, stderr=subprocess.STDOUT)
            return self.run_id

    def close_worker(self):
        if self.process is not None and self.process.poll() is None:
            if os.name == 'nt' and type(self.process.pid) is int:
                # venv launchers can own a child interpreter on Windows.
                # Terminate only this server's explicitly owned worker tree.
                result = subprocess.run(['taskkill.exe','/PID',str(self.process.pid),'/T','/F'],capture_output=True)
                if result.returncode and self.process.poll() is None:
                    reason=(result.stderr or result.stdout).decode(errors='replace').strip()
                    raise RuntimeError(f'Could not stop the owned training worker: {reason}')
            else:
                self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log_handle:
            self.log_handle.close()

    def reset(self):
        with self.lock:
            self.close_worker()
            self.process = self.run_id = self.state_path = self.log_handle = None


def read_points(root, n):
    # No numerical/model import is needed for immediate input preview.
    import struct
    f32=lambda v:struct.unpack('f',struct.pack('f',float(v)))[0]
    with (root/'inputs'/'datasets'/f'paramrep_2d_{n}.csv').open(newline='') as handle:
        return [dict(point_id=r['point_id'],group=r['group'],x=f32(r['math_score']),y=f32(r['english_score'])) for r in csv.DictReader(handle)]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def respond(self, code, value, content_type='application/json; charset=utf-8'):
        payload = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == '/':
            self.respond(200, (HERE / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        elif self.path == '/api/status':
            with self.server.lock:
                self.respond(200, self.server.status())
        elif self.path == '/api/info':
            self.respond(200,dict(replay_version=1,features=['pairs','frames','reset']))
        elif urlsplit(self.path).path in ('/api/pairs','/api/frames'):
            route=urlsplit(self.path)
            query=parse_qs(route.query,keep_blank_values=True)
            expected={'run_id'} if route.path=='/api/pairs' else {'run_id','after'}
            if set(query)!=expected or any(len(v)!=1 for v in query.values()):
                return self.respond(400,dict(error='Invalid replay query.'))
            try:
                after=int(query.get('after',['-1'])[0])
                if not -1<=after<=450:raise ValueError()
            except ValueError:
                return self.respond(400,dict(error='after must be an epoch from -1 to 450.'))
            with self.server.lock:
                if self.server.run_id is None or query['run_id'][0]!=self.server.run_id:
                    return self.respond(409,dict(error='Run changed; reload its status.'))
                directory=self.server.state_path.parent
                if route.path=='/api/pairs':
                    try:document=json.loads((directory/'pairs.json').read_text(encoding='utf-8'))
                    except (OSError,ValueError):return self.respond(202,dict(ready=False,run_id=self.server.run_id))
                    return self.respond(200,dict(ready=True,run_id=self.server.run_id,trace=document))
                frames=[]
                try:
                    with (directory/'frames.jsonl').open(encoding='utf-8') as handle:
                        for line in handle:
                            if not line.endswith('\n'):break
                            frame=json.loads(line)
                            if frame['epoch']>after:
                                frames.append(frame)
                                if len(frames)==128:break
                except (OSError,ValueError):pass
                self.respond(200,dict(run_id=self.server.run_id,frames=frames))
        else:
            self.respond(404, dict(error='Not found.'))

    def do_POST(self):
        if self.path not in ('/api/start','/api/reset'):
            return self.respond(404, dict(error='Not found.'))
        origin = self.headers.get('Origin')
        allowed_origin = f'http://127.0.0.1:{self.server.server_port}'
        if origin is not None and origin != allowed_origin:
            return self.respond(403, dict(error='Open the viewer at its printed 127.0.0.1 address.'))
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            return self.respond(415, dict(error='application/json is required.'))
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 4096:
                raise ValueError('Invalid request size.')
            request = json.loads(self.rfile.read(size))
            if not isinstance(request, dict):
                raise ValueError('Expected an object.')
            if self.path=='/api/reset':
                if request:raise ValueError('Reset expects an empty object.')
                self.server.reset()
                return self.respond(200,dict(status='idle'))
            run_id = self.server.start(request)
            if run_id is None:
                return self.respond(409, dict(error='Training is already running. Wait until it finishes.'))
            self.respond(202, dict(run_id=run_id, status='preparing'))
        except (ValueError, TypeError) as exc:
            self.respond(400, dict(error=str(exc)))
        except RuntimeError as exc:
            self.respond(503,dict(error=str(exc)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE.parents[1])
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = TrainingServer(args.root.resolve(), args.port)
    print(f'Open http://127.0.0.1:{server.server_port} (local to this computer only).', flush=True)
    print(f'Run logs and exact epoch coordinates: {server.run_directory}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close_worker()
        server.server_close()


if __name__ == '__main__':
    main()
