"""Evaluate an APPL library on the paper's 96 frozen APPL episodes, then compare.

The episode list is the published APPL set in results/exp2/episodes.jsonl (motion 40,
task 40, composition 16), so every new outcome pairs with a published one by case and
initial-state hash. Each episode is one `python -m appl evaluate --method appl` run in the
locked Exp2 environment, in <root>/<suite>/<task>/<case>. Finished episodes are skipped;
an episode directory without result.json is reported, never re-run automatically.

  run:      python scripts/exp2_evaluate.py run --root runs/exp2_eval/skillforge \
                --construction-root runs/exp2_skillforge --api-config codex.json --gpu 0 --gpu 3
  summary:  python scripts/exp2_evaluate.py summary --root runs/exp2_eval/skillforge [--root ...]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'environments/exp2/pixi.toml'
SUITES = ('motion', 'task', 'composition')


def published():
    rows = [json.loads(line) for line in (ROOT/'results/exp2/episodes.jsonl').read_text().splitlines() if line]
    return [r for r in rows if r['method'] == 'appl']


def run(args):
    root = Path(args.root).resolve()
    gpus = itertools.cycle(args.gpu)
    lock = threading.Lock()
    episodes = [e for e in published() if (not args.suites or e['suite'] in args.suites)
                and (not args.tasks or e['task'] in args.tasks)]
    if args.construction_root:
        # An episode against an unfinished library fails and is never re-run, so skip such tasks.
        ready = {task for task in {e['task'] for e in episodes} if library_ready(Path(args.construction_root)/task)}
        for task in sorted({e['task'] for e in episodes} - ready):
            print(json.dumps(dict(skipped_task=task, reason='construction library incomplete')), flush=True)
        episodes = [e for e in episodes if e['task'] in ready]

    def one(episode):
        out = root/episode['suite']/episode['task']/episode['case_id']
        if (out/'result.json').exists():
            return 'done'
        if out.exists():
            print(json.dumps(dict(skipped=str(out.relative_to(root)), reason='earlier attempt has no result')), flush=True)
            return 'incomplete'
        with lock:
            gpu = next(gpus)
        cmd = ['pixi', 'run', '--manifest-path', str(MANIFEST), '--locked', 'python', '-m', 'appl', 'evaluate',
               '--task', episode['task'], '--method', 'appl', '--suite', episode['suite'], '--case', episode['case_id'],
               '--gpu', str(gpu), '--output', str(out)]
        if args.construction_root:
            cmd += ['--construction-run', str(Path(args.construction_root).resolve()/episode['task'])]
        if args.api_config:
            cmd += ['--api-config', str(Path(args.api_config).resolve())]
        log = root/'logs'/f"{episode['suite']}_{episode['task']}_{episode['case_id']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with log.open('a') as stream:
            code = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
        result = json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else None
        print(json.dumps(dict(episode=f"{episode['suite']}/{episode['task']}/{episode['case_id']}", code=code,
                              seconds=round(time.time()-started),
                              success=None if result is None else result['success'],
                              first_attainment=None if result is None else result.get('first_attainment_success'))),
              flush=True)
        return 'ran'

    with ThreadPoolExecutor(args.sessions) as pool:
        list(pool.map(one, episodes))


def library_ready(run):
    manifest = run/'cut/manifest.json'
    if not manifest.exists():
        return False
    skills = [row['skill_id'] for row in json.loads(manifest.read_text())['datasets']]
    return all((run/'library'/skill/'receipt.json').exists() for skill in skills)


def outcomes(root):
    rows = {}
    for episode in published():
        path = Path(root)/episode['suite']/episode['task']/episode['case_id']/'result.json'
        if path.exists():
            result = json.loads(path.read_text())
            if result['paired_initial_sha256'] != episode['paired_initial_sha256']:
                raise ValueError('Initial state differs from the published pair: '+str(path))
            rows[(episode['suite'], episode['task'], episode['case_id'])] = result
    return rows


def summary(args):
    pub = {(e['suite'], e['task'], e['case_id']): e for e in published()}
    columns = [('published APPL', {k: dict(first=v['success'], final=v['success']) for k, v in pub.items()})]
    for root in args.root:
        rows = outcomes(root)
        columns.append((Path(root).name, {k: dict(first=bool(r.get('first_attainment_success', r['success'])),
                                                   final=bool(r['success'])) for k, r in rows.items()}))
    lines = ['| condition | ' + ' | '.join(f'{s} (first / final)' for s in SUITES) + ' | episodes |',
             '| --- | ' + ' | '.join('---' for _ in SUITES) + ' | --- |']
    for name, table in columns:
        cells = []
        for suite in SUITES:
            keys = [k for k in pub if k[0] == suite]
            done = [table[k] for k in keys if k in table]
            cells.append(f"{sum(d['first'] for d in done)} / {sum(d['final'] for d in done)} of {len(done)}"
                         + ('' if len(done) == len(keys) else f' (of {len(keys)})'))
        lines.append(f'| {name} | ' + ' | '.join(cells) + f' | {len(table)} |')
    print('\n'.join(lines))
    if len(columns) > 1:
        for name, table in columns[1:]:
            for other_name, other in columns:
                if other is table:
                    continue
                shared = [k for k in table if k in other]
                wins = sum(table[k]['first'] and not other[k]['first'] for k in shared)
                losses = sum(other[k]['first'] and not table[k]['first'] for k in shared)
                print(f'{name} vs {other_name}: {len(shared)} paired episodes, '
                      f'{wins} only {name} succeeds, {losses} only {other_name} succeeds (first attainment)')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--root', required=True)
    r.add_argument('--construction-root', help='directory holding one construction run per task')
    r.add_argument('--api-config', help='JSON API settings, e.g. {"backend": "codex", ...}')
    r.add_argument('--gpu', type=int, action='append', required=True)
    r.add_argument('--sessions', type=int, default=4)
    r.add_argument('--suites', nargs='*', choices=SUITES)
    r.add_argument('--tasks', nargs='*', help='evaluate only these tasks')
    s = sub.add_parser('summary')
    s.add_argument('--root', action='append', default=[])
    args = parser.parse_args()
    run(args) if args.command == 'run' else summary(args)


if __name__ == '__main__':
    main()
