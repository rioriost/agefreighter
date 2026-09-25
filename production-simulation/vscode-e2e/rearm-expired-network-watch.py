#!/usr/bin/env python3
"""One-time R2 rearming after the first no-load expiry; preserves evidence.

Operator must first recheck ARM ownership/governance and the existing budget.
Never invoke after migration begins. This is not product auto-recovery.
"""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess


def command(*args):
    return subprocess.run(args, capture_output=True, text=True, timeout=20)


def main():
    assert __debug__ and os.geteuid() == 0
    root = Path('/var/lib/agefreighter/workflows/b2c7214e-83f5-4613-b378-98d36e0cd97d')
    boot = '3c50ec8b-3ca1-4997-907c-0ed6e2c013dc'
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == boot
    previous = 'af-network-watch-20260916-b2c7214e.service'
    state = command('systemctl', 'show', previous, '-p', 'ActiveState', '-p', 'Result', '-p', 'ExecMainStatus')
    assert state.returncode == 0
    assert dict(line.split('=', 1) for line in state.stdout.splitlines()) == {
        'ActiveState': 'inactive', 'Result': 'success', 'ExecMainStatus': '0'}
    for path in root.glob('*/state.json'):
        item = json.loads(path.read_text())
        assert item.get('action') not in ('migrate-source', 'migrate-csv', 'resume-migration')
        assert item.get('phase') not in ('accepted', 'running')
    rules = command('iptables-save')
    assert rules.returncode == 0 and 'af-network-' not in rules.stdout
    assert list(root.rglob('qualification-network-*.json')) == [root / 'qualification-network-armed.json']
    armed = root / 'qualification-network-armed.json'
    log = root / 'qualification-network-watch.log'
    expected = '9dbfffd9b4b42a264fa2a6218a65ee766cba5bd59144942a11c0a1b86fb8e80e'
    assert hashlib.sha256(armed.read_bytes()).hexdigest() == expected
    assert json.loads(log.read_text().strip()) == {'expired': True, 'migrationStarted': False}
    now = dt.datetime.now(dt.timezone.utc)
    assert now > dt.datetime.fromisoformat(json.loads(armed.read_text())['deadline'].replace('Z', '+00:00'))
    deadline = now + dt.timedelta(minutes=15)
    assert deadline < dt.datetime(2026, 9, 17, 7, tzinfo=dt.timezone.utc)
    stat = os.statvfs(root)
    disk = 100 * (stat.f_blocks - stat.f_bfree) / stat.f_blocks
    assert disk < 80
    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    assert int(mem['SwapTotal'].split()[0]) == int(mem['SwapFree'].split()[0])
    assert int(dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())['oom_kill']) == 0
    assert {a[4][0] for a in socket.getaddrinfo('neo4j526.azn526.internal', 7687, type=socket.SOCK_STREAM)} == {'10.246.5.4'}
    tools = root / 'qualification-network-tools'
    observer_sha = 'eff22e0f7770c40343addba53328b2dbed548065db7a6182f071aff2beac2035'
    assert hashlib.sha256((tools / 'observe-recovery-guest.py').read_bytes()).hexdigest() == observer_sha
    assert hashlib.sha256((tools / 'await-network-load.py').read_bytes()).hexdigest() == '128a09d1e5f0672dd3c20143f4b1ff240c3bdb149e1f342d2eba77eaf66d3735'
    archive = root / 'qualification-expired-arming1'
    archive.mkdir(mode=0o700)  # create-only; ambiguous retries fail closed
    for path in (armed, log):
        path.rename(archive / path.name)
    assert hashlib.sha256((archive / armed.name).read_bytes()).hexdigest() == expected
    manifest = {'previousArmingSHA256': expected, 'previousLogSHA256': hashlib.sha256((archive / log.name).read_bytes()).hexdigest(),
                'rearmedAt': now.isoformat(), 'deadline': deadline.isoformat(), 'diskUsedPercent': disk,
                'noPreviousMigration': True, 'previousUnit': previous}
    with (archive / 'rearm-manifest.json').open('x') as output:
        os.chmod(output.name, 0o600)
        json.dump(manifest, output)
        output.flush()
        os.fsync(output.fileno())
    for directory in (archive, root):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(descriptor)
        os.close(descriptor)
    unit = 'af-network-watch-20260916-b2c7214e-arm2'
    result = command('systemd-run', '--quiet', '--unit=' + unit, '--property=RuntimeMaxSec=960',
                     '--property=MemoryMax=268435456', '--property=MemorySwapMax=0', '--property=UMask=0077',
                     '--property=StandardOutput=append:' + str(root / 'qualification-network-watch-arm2.log'),
                     '--property=StandardError=append:' + str(root / 'qualification-network-watch-arm2.log'),
                     '/usr/bin/python3', str(tools / 'await-network-load.py'), '--boot', boot,
                     '--deadline', deadline.isoformat(), '--observer-sha256', observer_sha)
    assert result.returncode == 0
    assert command('systemctl', 'is-active', unit).stdout.strip() == 'active'
    print(json.dumps({**manifest, 'unit': unit, 'active': True}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        import traceback
        frame = traceback.extract_tb(error.__traceback__)[-1]
        print(json.dumps({'stopped': True, 'errorType': type(error).__name__, 'line': frame.lineno}), flush=True)
        raise SystemExit(1)
