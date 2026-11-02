"""HTTP/control unit tests; real model/browser checks live in verification.json."""
import json
import os
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from server import TrainingServer


class TestServer(unittest.TestCase):
    def setUp(self):
        self.server = TrainingServer(Path(os.environ.get('PARAMREP_TEST_ROOT',str(Path(__file__).resolve().parents[2]))), 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.close_worker()
        self.server.server_close()
        self.thread.join()

    def post(self, payload, **headers):
        request = Request(self.url + '/api/start', data=json.dumps(payload).encode(),
                          headers={'Content-Type': 'application/json', **headers}, method='POST')
        try:
            response = urlopen(request)
        except HTTPError as error:
            response = error
        return response.status, json.load(response)

    def test_idle_and_page(self):
        with urlopen(self.url + '/api/status') as response:
            self.assertEqual(json.load(response), {'status': 'idle'})
        with urlopen(self.url) as response:
            page = response.read().decode()
        self.assertIn('Start training', page)
        self.assertIn('not an interpolated animation', page)

    def test_validation_never_starts_worker(self):
        bad = [None, [], {}, {'variant': 'bad', 'n': 36, 'seed': 42},
               {'variant': 'baseline', 'n': 37, 'seed': 42},
               {'variant': 'baseline', 'n': 36, 'seed': True},
               {'variant': 'baseline', 'n': 36, 'seed': -1},
               {'variant': 'baseline', 'n': 36, 'seed': 2**31},
               {'variant': 'baseline', 'n': 36, 'seed': 42, 'epochs': 1}]
        with patch('server.subprocess.Popen') as process:
            for payload in bad:
                with self.subTest(payload=payload):
                    code, value = self.post(payload)
                    self.assertEqual(code, 400)
                    self.assertIn('error', value)
            process.assert_not_called()

    def test_reject_cross_origin(self):
        code, _ = self.post({'variant': 'baseline', 'n': 36, 'seed': 42}, Origin='https://example.com')
        self.assertEqual(code, 403)

    def test_single_active_worker(self):
        # Mock process lifecycle only; no mock coordinates or outputs are shown.
        process = Mock()
        process.poll.return_value = None
        with patch('server.subprocess.Popen', return_value=process) as create:
            code, value = self.post({'variant': 'baseline', 'n': 36, 'seed': 42})
            self.assertEqual(code, 202)
            first_id = value['run_id']
            code, value = self.post({'variant': 'modified', 'n': 50, 'seed': 4})
            self.assertEqual(code, 409)
            self.assertEqual(create.call_count, 1)
            self.assertEqual(self.server.run_id, first_id)
            process.poll.return_value = 0
            code, value = self.post({'variant': 'modified', 'n': 50, 'seed': 4})
            self.assertEqual(code, 202)
            self.assertNotEqual(first_id, value['run_id'])
            self.assertEqual(create.call_count, 2)

    def test_missing_and_traversal_paths(self):
        for path in ['/nope', '/../worker.py', '/api/start']:
            with self.subTest(path=path):
                with self.assertRaises(HTTPError) as error:
                    urlopen(self.url + path)
                self.assertEqual(error.exception.code, 404)

    def test_completion_waits_for_worker_exit(self):
        process = Mock()
        process.poll.return_value = None
        self.server.process = process
        self.server.run_id = 'unit-finalizing'
        self.server.state_path = self.server.run_directory / 'unit-state.json'
        self.server.state_path.write_text(json.dumps({'status': 'completed', 'epoch': 450}))
        self.server.log_handle = (self.server.run_directory / 'unit-log.txt').open('w')
        self.assertEqual(self.server.status()['status'], 'finalizing')
        process.poll.return_value = 0
        self.assertEqual(self.server.status()['status'], 'completed')

    def test_immediate_state_exists_before_worker_launch(self):
        process=Mock();process.poll.return_value=0
        def launch(command,**kwargs):
            state=json.loads(Path(command[command.index('--state')+1]).read_text())
            self.assertEqual(state['stage'],'starting_worker')
            self.assertEqual(len(state['points']),36)
            self.assertIsNone(state['coordinates'])
            return process
        with patch('server.subprocess.Popen',side_effect=launch):
            self.assertEqual(self.post({'variant':'baseline','n':36,'seed':42})[0],202)

    def test_replay_endpoints_and_reset(self):
        with urlopen(self.url+'/api/info') as response:self.assertEqual(json.load(response)['replay_version'],1)
        process=Mock();process.poll.return_value=None
        with patch('server.subprocess.Popen',return_value=process):
            self.assertEqual(self.post({'variant':'baseline','n':36,'seed':42})[0],202)
        run=self.server.run_id;directory=self.server.state_path.parent
        with urlopen(self.url+f'/api/pairs?run_id={run}') as response:
            self.assertEqual(response.status,202);self.assertFalse(json.load(response)['ready'])
        (directory/'pairs.json').write_text(json.dumps({'schema_version':1,'pairs':{'NN':[[0,1]]}}))
        with urlopen(self.url+f'/api/pairs?run_id={run}') as response:self.assertTrue(json.load(response)['ready'])
        (directory/'frames.jsonl').write_text(json.dumps({'epoch':0,'coordinates':[[1,2]]})+'\n'+json.dumps({'epoch':1,'coordinates':[[2,3]]})+'\n'+ '{"epoch":2')
        with urlopen(self.url+f'/api/frames?run_id={run}&after=-1') as response:self.assertEqual([f['epoch'] for f in json.load(response)['frames']],[0,1])
        with urlopen(self.url+f'/api/frames?run_id={run}&after=0') as response:self.assertEqual([f['epoch'] for f in json.load(response)['frames']],[1])
        for route,code in [('/api/frames?run_id=wrong&after=0',409),(f'/api/frames?run_id={run}&after=-2',400),('/api/pairs',400)]:
            with self.assertRaises(HTTPError) as error:urlopen(self.url+route)
            self.assertEqual(error.exception.code,code)
        request=Request(self.url+'/api/reset',data=b'{}',headers={'Content-Type':'application/json'},method='POST')
        with urlopen(request) as response:self.assertEqual(json.load(response),{'status':'idle'})
        process.terminate.assert_called_once();self.assertEqual(self.server.status(),{'status':'idle'})
        self.assertTrue((directory/'frames.jsonl').exists(),'reset preserves recorded files')

    def test_reset_rejects_cross_origin_and_invalid_body(self):
        for body,origin,code in [(b'{}','https://example.com',403),(b'{"n":36}',None,400)]:
            headers={'Content-Type':'application/json'}
            if origin:headers['Origin']=origin
            request=Request(self.url+'/api/reset',data=body,headers=headers,method='POST')
            with self.assertRaises(HTTPError) as error:urlopen(request)
            self.assertEqual(error.exception.code,code)

    def test_transient_state_read_keeps_replay_protocol(self):
        process=Mock();process.poll.return_value=None
        self.server.process=process;self.server.run_id='test-transient';self.server.state_path=self.server.run_directory/'state.json'
        self.server.state_path.write_text('{}')
        with patch.object(Path,'read_text',side_effect=PermissionError('transient reader')):
            self.assertEqual(self.server.status()['replay_version'],1)
        process.poll.return_value=0


if __name__ == '__main__':
    unittest.main()
