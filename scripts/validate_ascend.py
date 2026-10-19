import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

workspace = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(workspace / 'src'))
from ntops_lab import runnable_operators

root = Path(os.environ.get('NTOPS_VALIDATION_OUTPUT', '/tmp/ntops-final-parallel-validation'))
root.mkdir(exist_ok=True)
records = sorted(runnable_operators(), key=lambda item: item.op)
workers = []
devices = tuple(int(value) for value in os.environ.get('NTOPS_VALIDATION_DEVICES', '0,1,5,6,7').split(','))
for slot, device in enumerate(devices):
    selected = records[slot::len(devices)]
    targets = [f'tests/test_operator_gpu_validation.py::test_operator_matches_pytorch[{record.op}]' for record in selected]
    env = dict(os.environ, NTOPS_RUN_OPERATOR_VALIDATION='1', ASCEND_RT_VISIBLE_DEVICES=str(device))
    log = (root / f'npu-{slot+1}.log').open('w')
    command = [sys.executable, '-m', 'pytest', *targets, '--tb=short', f'--junitxml={root}/npu-{slot+1}.xml']
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=workspace)
    workers.append((slot+1, process, log))
print(f'Started validation on NPU devices {devices}.', flush=True)
while any(process.poll() is None for _, process, _ in workers):
    time.sleep(5)
    done = [slot for slot, process, _ in workers if process.poll() is not None]
    if done:
        print(f'Completed devices: {done}', flush=True)
        time.sleep(20)
results = []
for slot, process, log in workers:
    log.close()
    xml = root / f'npu-{slot}.xml'
    if not xml.exists():
        results.append({'device': slot, 'status': 'runner_error', 'exit_code': process.returncode})
        continue
    for case in ET.parse(xml).iter('testcase'):
        failure = case.find('failure')
        error = case.find('error')
        skipped = case.find('skipped')
        status = 'failed' if failure is not None or error is not None else 'skipped' if skipped is not None else 'passed'
        result = {'name': case.attrib['name'], 'status': status, 'seconds': case.attrib.get('time')}
        if failure is not None or error is not None:
            detail = failure if failure is not None else error
            result['message'] = detail.attrib.get('message', '')
            result['traceback'] = detail.text
        results.append(result)
(root / 'results.json').write_text(json.dumps(results, indent=2))
print(json.dumps({'passed': sum(r['status']=='passed' for r in results), 'failed': sum(r['status']=='failed' for r in results), 'skipped': sum(r['status']=='skipped' for r in results), 'total': len(results)}), flush=True)

sys.exit(0 if len(results) == len(records) and all(r["status"] == "passed" for r in results) else 1)
