"""Run the full Exp2 construction sequence for new libraries, resuming completed stages.

For every task: init (once), Cut, then for every Cut skill: design (h01-h03), train and
verify h01-h03, refine (h04), train and verify h04, finalize. Stages are the
`python -m appl.construction` commands of experiments/exp2/construction.md, run through the
locked Exp2 environment. A finished stage is recognized by its output and skipped; a failed
stage stops that skill (or task) and is recorded in <run>/driver_failures.jsonl, without an
automatic retry; the one exception is a GPU job that APPL's launcher refused because the
selected GPU had less than 2 GiB free (it never started), which is retried every five minutes.
A stage that an earlier driver left running is waited for, never launched twice. Agent
sessions and GPU jobs run with bounded concurrency. Disk: the 2-update checkpoint of a finished
`check_policy` job (about 365 MB, never read again; the check result is kept in the journal)
is deleted, and no stage starts while the user's quota has less than --min-free-gb left.

Restart rule (both arms): a design, revision or final-guidance session that ends without
finishing its stage is restarted once from scratch. The ended session's files are moved to
<resets>/<arm>/<task>/<skill>/<stage>_attempt1 and the restart is logged in
<resets>/resets.jsonl; a second failure is final.

Usage:
  python scripts/exp2_construct.py --root runs/exp2_skillforge --prompt-system skillforge \
      --backend codex [--model gpt-6-sol] --gpu drawer_exchange=1 --gpu buffer_swap=3 ... \
      [--sessions 4] [--per-gpu 3]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'environments/exp2/pixi.toml'
FULL_GPU = 'Selected GPU has less than 2 GiB free'
TASKS = ('drawer_exchange', 'buffer_swap', 'constrained_retrieve_store', 'covered_peg_assembly',
         'granular_pour_return')


def quota_free_gb():
    """Space left under the user's disk quota (falls back to the filesystem's free space)."""
    try:
        out = subprocess.run(['quota', '-w', '-p'], capture_output=True, text=True).stdout.split('\n')
        row = next(line.split() for line in out if line.strip().startswith('/dev/'))
        used, limit = int(row[1].rstrip('*')), int(row[3])
        if limit:
            return (limit - used) / 2**20
    except (OSError, StopIteration, ValueError, IndexError):
        pass
    stat = os.statvfs(ROOT)
    return stat.f_bavail * stat.f_frsize / 2**30


def janitor(root):
    """Delete checkpoints of finished check_policy jobs (their results stay in the journal)."""
    while True:
        for checkpoint in Path(root).glob('*/policies/*/checks/*/last.pt'):
            if (checkpoint.parent/'result.json').exists():
                checkpoint.unlink(missing_ok=True)
        for checkpoint in Path(root).glob('*/refinement/policies/*/checks/*/last.pt'):
            if (checkpoint.parent/'result.json').exists():
                checkpoint.unlink(missing_ok=True)
        time.sleep(120)


class Driver:
    def __init__(self, args):
        self.args = args
        self.sessions = threading.BoundedSemaphore(args.sessions)
        self.gpus = {}
        self.lock = threading.Lock()

    def gpu(self, index):
        with self.lock:
            slots = self.args.gpu_slots.get(index, self.args.per_gpu)
            return self.gpus.setdefault(index, threading.BoundedSemaphore(slots))

    def run(self, run, label, stage, *extra, slot, marker):
        """Run one stage unless ``marker`` shows it is done; never run a stage twice at once."""
        signature = ' '.join(['appl.construction', stage, '--run', str(run), *extra])
        while self.running(signature):
            time.sleep(60)
        if marker.exists():
            return
        while quota_free_gb() < self.args.min_free_gb:
            self.note(run, 'disk_wait', label=label, free_gb=round(quota_free_gb(), 1))
            time.sleep(300)
        logs = run/'driver_logs'
        logs.mkdir(parents=True, exist_ok=True)
        cmd = ['pixi', 'run', '--manifest-path', str(MANIFEST), '--locked', 'python', '-m', *signature.split(' ')]
        while True:
            with slot:
                started = time.time()
                self.note(run, 'start', label=label)
                with (logs/(label+'.log')).open('a') as stream:
                    code = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
                self.note(run, 'end', label=label, code=code, seconds=round(time.time()-started))
            if code and stage in ('train', 'verify') and self.gpu_was_full(logs/(label+'.log')):
                self.note(run, 'gpu_full_retry', label=label)
                time.sleep(300)
                continue
            break
        if code:
            with (run/'driver_failures.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(label=label, code=code, time=time.time()))+'\n')
            raise RuntimeError(f'{run.name}: {label} failed ({code}); see driver_logs/{label}.log')

    @staticmethod
    def gpu_was_full(log):
        """The launcher's refusal is written to the worker log the stage log points to."""
        tail = log.read_text()[-4000:]
        texts = [tail] + [Path(p).read_text()[-4000:] for p in re.findall(r'inspect (\S+worker\.log)', tail)
                          if Path(p).exists()]
        return any(FULL_GPU in text for text in texts)

    @staticmethod
    def running(signature):
        listing = subprocess.run(['pgrep', '-af', 'appl.construction'], capture_output=True, text=True).stdout
        return any(line.endswith(signature) or signature+' ' in line for line in listing.splitlines())

    def note(self, run, kind, **payload):
        line = json.dumps(dict(task=run.name, kind=kind, time=time.strftime('%H:%M:%S'), **payload))
        print(line, flush=True)

    def task(self, task):
        run = (Path(self.args.root)/task).resolve()
        gpu = self.args.gpu[task]
        if not (run/'configuration.json').exists():
            subprocess.run(['pixi', 'run', '--manifest-path', str(MANIFEST), '--locked', 'python', '-m',
                            'appl.construction', 'init', '--task', task, '--output', str(run), '--gpu', str(gpu),
                            '--backend', self.args.backend, '--prompt-system', self.args.prompt_system,
                            *(['--model', self.args.model] if self.args.model else [])],
                           cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        self.run(run, 'cut', 'cut', slot=self.sessions, marker=run/'cut/manifest.json')
        skills = [row['skill_id'] for row in json.loads((run/'cut/manifest.json').read_text())['datasets']]
        with ThreadPoolExecutor(len(skills)) as pool:
            outcomes = list(pool.map(lambda skill: self.skill(run, gpu, skill), skills))
        failed = [s for s, ok in zip(skills, outcomes) if not ok]
        self.note(run, 'task_done', failed_skills=failed)

    def skill(self, run, gpu, skill):
        try:
            self.agent_stage(run, skill, 'design', run/'skills'/skill/'submission.json',
                             [run/'skills'/skill, *sorted((run/'policies').glob(f'{skill}__h0[123]'))])
            self.policies(run, gpu, [f'{skill}__h0{i}' for i in (1, 2, 3)])
            self.agent_stage(run, skill, 'refine', run/'refinement/skills'/skill/'submission.json',
                             [run/'refinement/skills'/skill, run/'refinement/policies'/f'{skill}__h04'])
            self.policies(run, gpu, [f'{skill}__h04'])
            self.agent_stage(run, skill, 'finalize', run/'library'/skill/'receipt.json', [run/'library'/skill])
            return True
        except Exception as error:  # one skill's failure does not stop the others
            self.note(run, 'skill_failed', skill=skill, error=str(error))
            return False

    def agent_stage(self, run, skill, stage, marker, owned):
        """Run an agent stage; restart it once from scratch if it ends without finishing."""
        label = f'{stage}_{skill}'
        arm = run.parent.name
        try:
            self.run(run, label, stage, '--skill', skill, slot=self.sessions, marker=marker)
            return
        except RuntimeError as error:
            resets = Path(self.args.resets)
            log = resets/'resets.jsonl'
            done = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            if any(r.get('kind') == 'agent_session_restart' and r.get('arm') == arm and r.get('task') == run.name
                   and r.get('skill') == skill and r.get('stage') == stage for r in done):
                raise
            # Locked rows never move under a live session: the stage process has exited here.
            dest = resets/arm/run.name/skill/f'{stage}_attempt1'
            dest.mkdir(parents=True, exist_ok=False)
            moved = []
            for path in owned:
                if path.exists():
                    path.rename(dest/path.name)
                    moved.append(str(path))
            with self.lock, log.open('a') as stream:
                stream.write(json.dumps(dict(time=time.strftime('%Y-%m-%dT%H:%M:%S'), arm=arm, task=run.name,
                    kind='agent_session_restart', skill=skill, stage=stage, error=str(error)[:300], moved=moved,
                    reason='session ended without finishing its stage; restarted once (standing rule, both arms)'))+'\n')
            self.note(run, 'agent_session_restart', label=label)
        self.run(run, label, stage, '--skill', skill, slot=self.sessions, marker=marker)

    def policies(self, run, gpu, identifiers):
        def one(policy):
            base = run/('refinement' if policy.endswith('__h04') else '')/'policies'/policy
            self.run(run, f'train_{policy}', 'train', '--policy', policy, slot=self.gpu(gpu),
                     marker=base/'training/result.json')
            self.run(run, f'verify_{policy}', 'verify', '--policy', policy, slot=self.gpu(gpu),
                     marker=run/'verification'/policy/'receipt.json')
        with ThreadPoolExecutor(len(identifiers)) as pool:
            for future in [pool.submit(one, p) for p in identifiers]:
                future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', required=True)
    parser.add_argument('--prompt-system', choices=('appl', 'skillforge'), required=True)
    parser.add_argument('--backend', choices=('responses', 'codex'), required=True)
    parser.add_argument('--model', help='model for every agent session (default: the paper model)')
    parser.add_argument('--gpu', action='append', required=True, help='task=index')
    parser.add_argument('--tasks', nargs='*', default=list(TASKS), choices=TASKS)
    parser.add_argument('--sessions', type=int, default=4, help='concurrent agent sessions')
    parser.add_argument('--per-gpu', type=int, default=3, help='concurrent training/verification jobs per GPU')
    parser.add_argument('--gpu-slots', action='append', default=[], help='index=jobs, overrides --per-gpu for one GPU')
    parser.add_argument('--min-free-gb', type=float, default=15, help='start no stage below this much quota left')
    parser.add_argument('--resets', help='archive and log of restarted sessions (default: <root parent>_resets)')
    args = parser.parse_args()
    args.gpu = {k: int(v) for k, v in (item.split('=') for item in args.gpu)}
    args.gpu_slots = {int(k): int(v) for k, v in (item.split('=') for item in args.gpu_slots)}
    root = Path(args.root).resolve()
    args.resets = args.resets or str(root.parent.with_name(root.parent.name+'_resets'))
    missing = set(args.tasks) - set(args.gpu)
    if missing:
        parser.error('No GPU for: '+', '.join(sorted(missing)))
    threading.Thread(target=janitor, args=(args.root,), daemon=True).start()
    driver = Driver(args)
    with ThreadPoolExecutor(len(args.tasks)) as pool:
        list(pool.map(driver.task, args.tasks))


if __name__ == '__main__':
    main()
