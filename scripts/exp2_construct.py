"""Run the full Exp2 construction sequence for new libraries, resuming completed stages.

For every task: init (once), Cut, then for every Cut skill: design (h01-h03), train and
verify h01-h03, refine (h04), train and verify h04, finalize. Stages are the
`python -m appl.construction` commands of experiments/exp2/construction.md, run through the
locked Exp2 environment. A finished stage is recognized by its output and skipped; a failed
stage stops that skill (or task) and is recorded in <run>/driver_failures.jsonl, without an
automatic retry. Agent sessions and GPU jobs run with bounded concurrency.

Usage:
  python scripts/exp2_construct.py --root runs/exp2_skillforge --prompt-system skillforge \
      --backend codex [--model gpt-6-sol] --gpu drawer_exchange=1 --gpu buffer_swap=3 ... \
      [--sessions 4] [--per-gpu 3]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'environments/exp2/pixi.toml'
TASKS = ('drawer_exchange', 'buffer_swap', 'constrained_retrieve_store', 'covered_peg_assembly',
         'granular_pour_return')


class Driver:
    def __init__(self, args):
        self.args = args
        self.sessions = threading.BoundedSemaphore(args.sessions)
        self.gpus = {}
        self.lock = threading.Lock()

    def gpu(self, index):
        with self.lock:
            return self.gpus.setdefault(index, threading.BoundedSemaphore(self.args.per_gpu))

    def run(self, run, label, stage, *extra, slot):
        logs = run/'driver_logs'
        logs.mkdir(parents=True, exist_ok=True)
        cmd = ['pixi', 'run', '--manifest-path', str(MANIFEST), '--locked', 'python', '-m',
               'appl.construction', stage, '--run', str(run), *extra]
        with slot:
            started = time.time()
            self.note(run, 'start', label=label)
            with (logs/(label+'.log')).open('a') as stream:
                code = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
            self.note(run, 'end', label=label, code=code, seconds=round(time.time()-started))
        if code:
            with (run/'driver_failures.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(label=label, code=code, time=time.time()))+'\n')
            raise RuntimeError(f'{run.name}: {label} failed ({code}); see driver_logs/{label}.log')

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
        if not (run/'cut/manifest.json').exists():
            self.run(run, 'cut', 'cut', slot=self.sessions)
        skills = [row['skill_id'] for row in json.loads((run/'cut/manifest.json').read_text())['datasets']]
        with ThreadPoolExecutor(len(skills)) as pool:
            outcomes = list(pool.map(lambda skill: self.skill(run, gpu, skill), skills))
        failed = [s for s, ok in zip(skills, outcomes) if not ok]
        self.note(run, 'task_done', failed_skills=failed)

    def skill(self, run, gpu, skill):
        try:
            if not (run/'skills'/skill/'submission.json').exists():
                self.run(run, f'design_{skill}', 'design', '--skill', skill, slot=self.sessions)
            self.policies(run, gpu, [f'{skill}__h0{i}' for i in (1, 2, 3)])
            if not (run/'refinement/skills'/skill/'submission.json').exists():
                self.run(run, f'refine_{skill}', 'refine', '--skill', skill, slot=self.sessions)
            self.policies(run, gpu, [f'{skill}__h04'])
            if not (run/'library'/skill/'receipt.json').exists():
                self.run(run, f'finalize_{skill}', 'finalize', '--skill', skill, slot=self.sessions)
            return True
        except Exception as error:  # one skill's failure does not stop the others
            self.note(run, 'skill_failed', skill=skill, error=str(error))
            return False

    def policies(self, run, gpu, identifiers):
        def one(policy):
            base = run/('refinement' if policy.endswith('__h04') else '')/'policies'/policy
            if not (base/'training/result.json').exists():
                self.run(run, f'train_{policy}', 'train', '--policy', policy, slot=self.gpu(gpu))
            if not (run/'verification'/policy/'receipt.json').exists():
                self.run(run, f'verify_{policy}', 'verify', '--policy', policy, slot=self.gpu(gpu))
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
    args = parser.parse_args()
    args.gpu = {k: int(v) for k, v in (item.split('=') for item in args.gpu)}
    missing = set(args.tasks) - set(args.gpu)
    if missing:
        parser.error('No GPU for: '+', '.join(sorted(missing)))
    driver = Driver(args)
    with ThreadPoolExecutor(len(args.tasks)) as pool:
        list(pool.map(driver.task, args.tasks))


if __name__ == '__main__':
    main()
