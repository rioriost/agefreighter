"""Execute exact generated B07 Python payload with inert OS/HTTP backends only."""
import builtins
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('b07_existing_fixtures',HERE/'test_b07_observe_inventory.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


class FinalWrapperTests(unittest.TestCase):
    def exercise(self,phase,mutate=None,wrong_vm=False):
        sha=hashlib.sha256((HERE/'b07-observe-inventory.py').read_bytes()).hexdigest()
        body=f.builder.bodies(f.binding(),sha,f.obs.BASE_SHA256)[phase]['body']
        script=body['properties']['source']['script']
        source=script.split("AF_B07_OBSERVER'\n",1)[1].rsplit('\nAF_B07_OBSERVER',1)[0]
        backend=f.backend()
        if mutate:mutate(backend)
        calls=[]
        # The exact wrapper imports both pinned modules via exec. Replace only
        # their read-only OS boundary after base import, preserving guest_backend,
        # observer, process checks, serialization and the entire wrapper itself.
        def imported(code,namespace):
            builtins.exec(code,namespace)
            if code.co_filename=='b07_pinned_base':
                cls=namespace['ReadOnlyBackend']
                for name in ('read','run','link','pagesize','utc'):
                    method=getattr(backend,name)
                    setattr(cls,name,lambda self,*args,_method=method:_method(*args))
        class Response:
            status=200
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def geturl(self):return f.obs.IMDS
            def read(self,limit):
                self_limit=2049
                if limit!=self_limit:raise AssertionError('Unexpected IMDS read bound')
                return (f.obs.VM_ID+('wrong' if wrong_vm else '')).encode()
        class Opener:
            def open(self,request,timeout):
                calls.append((request.get_method(),request.full_url,timeout))
                if request.get_method()!='GET' or request.full_url!=f.obs.IMDS or timeout!=5:
                    raise AssertionError('Unexpected fake transport request')
                return Response()
        output=io.StringIO()
        with patch('sys.platform','linux'), patch('os.geteuid',return_value=0), \
             patch('urllib.request.build_opener',return_value=Opener()), \
             patch('subprocess.Popen',side_effect=AssertionError('Forbidden real subprocess')), \
             patch('os.open',side_effect=AssertionError('Forbidden real file read')), \
             patch('os.readlink',side_effect=AssertionError('Forbidden real symlink read')), \
             patch('socket.create_connection',side_effect=AssertionError('Forbidden real network')), \
             contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as stopped:
                builtins.exec(compile(source,'exact-b07-final-wrapper','exec'),{'exec':imported})
        self.assertNotIn('PRIVATE',output.getvalue())
        self.assertLessEqual(len(output.getvalue().encode()),4096)
        return stopped.exception.code,json.loads(output.getvalue()),backend,calls

    def test_exact_before_and_after_final_payload_succeed_with_bounded_reads(self):
        for phase in ('before','after'):
            with self.subTest(phase=phase):
                status,result,backend,calls=self.exercise(phase)
                self.assertEqual(status,0)
                self.assertTrue(result['activeInventoryProcessProven'])
                self.assertEqual(result['phase'],phase)
                self.assertEqual(result['bindingSHA256'],f.obs.binding_sha(f.binding()))
                self.assertEqual(len(calls),1)
                self.assertEqual(len(backend.commands),2)
                self.assertTrue(all(x[:2]==['/bin/systemctl','show'] for x in backend.commands))
                self.assertFalse(any(any(term in path for term in ('job.json','secrets.json','environ','cmdline','meminfo')) for path in backend.reads))

    def test_wrong_imds_identity_fails_before_any_os_observation(self):
        status,result,backend,calls=self.exercise('before',wrong_vm=True)
        self.assertEqual(status,2);self.assertFalse(result['activeInventoryProcessProven'])
        self.assertEqual(backend.reads,[]);self.assertEqual(backend.commands,[])
        self.assertEqual(len(calls),1)

    def test_secret_bearing_backend_error_is_redacted_without_false_success(self):
        def mutate(backend):
            def failure(*args):raise RuntimeError('PRIVATE error from fake OS')
            backend.read=failure
        status,result,_,calls=self.exercise('after',mutate)
        self.assertEqual(status,2);self.assertFalse(result['activeInventoryProcessProven'])
        self.assertEqual(result['error'],'observation-inconclusive')
        self.assertFalse(result['retryAuthorized']);self.assertEqual(len(calls),1)

    def test_worker_pid_reuse_fails_through_exact_wrapper(self):
        def mutate(backend):
            stats=iter([f.fixtures.stat(102,101,'200'),f.fixtures.stat(102,101,'201')])
            backend.files['/proc/102/stat']=lambda:next(stats)
        status,result,_,_=self.exercise('after',mutate)
        self.assertEqual(status,2);self.assertFalse(result['activeInventoryProcessProven'])


if __name__=='__main__':unittest.main()
