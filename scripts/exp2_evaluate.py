"""Evaluate an APPL library on the paper's 96 frozen APPL episodes, then compare.

The episode list is the published APPL set in results/exp2/episodes.jsonl (motion 40,
task 40, composition 16), so every new outcome pairs with a published one by case and
initial-state hash. Each episode is one `python -m appl evaluate --method appl` run in the
locked Exp2 environment, in <root>/<suite>/<task>/<case>. Finished episodes are skipped;
an episode directory without result.json is reported, never re-run automatically.

  run:      python scripts/exp2_evaluate.py run --root runs/exp2_eval/skillforge \
                --construction-root runs/exp2_skillforge --api-config codex.json --gpu 0 --gpu 3
  summary:  python scripts/exp2_evaluate.py summary --root runs/exp2_eval/skillforge [--root ...]
  compare:  python scripts/exp2_evaluate.py compare --root runs/exp2_eval/appl --root runs/exp2_eval/skillforge
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import math
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


AGENT_ENDED = ('budget exhausted', 'idle beyond')


def outcomes(root):
    """Completed episodes, plus episodes whose agent session ended without a result.

    APPL's agent loop ends an episode whose agent uses all its model requests with failure.json
    and no result file; under Codex a session can also stall after the agent's last action until
    the idle timeout. Physics stopped with the agent, and every step's goal predicate is in the
    trace, so such an episode is scored from it: first attainment if any step met the goal, and
    final success from the last recorded step. Episodes that ended for other reasons stay unscored.
    """
    rows = {}
    for episode in published():
        folder = Path(root)/episode['suite']/episode['task']/episode['case_id']
        key = (episode['suite'], episode['task'], episode['case_id'])
        if (folder/'result.json').exists():
            result = json.loads((folder/'result.json').read_text())
            if result['paired_initial_sha256'] != episode['paired_initial_sha256']:
                raise ValueError('Initial state differs from the published pair: '+str(folder))
            rows[key] = result
        elif (folder/'failure.json').exists():
            failure = json.loads((folder/'failure.json').read_text())
            reason = next((r for r in AGENT_ENDED if r in failure.get('error', '')), None)
            if reason is None or not (folder/'trace.jsonl').exists():
                continue
            first = last = None
            for line in (folder/'trace.jsonl').read_text().splitlines():
                event = json.loads(line)
                if 'metrics' not in event:
                    continue
                last = event
                if first is None and event['metrics'].get('success'):
                    first = event.get('step')
            rows[key] = dict(success=bool(last and last['metrics'].get('success')),
                             first_attainment_success=first is not None, first_success_step=first,
                             status='request_budget_exhausted' if reason == 'budget exhausted' else 'session_stalled',
                             steps=failure.get('physical_steps'))
    return rows


def incomplete(root):
    return sorted(str(p.parent.relative_to(root)) for p in Path(root).glob('*/*/*/failure.json')
                  if not any(r in json.loads(p.read_text()).get('error', '') for r in AGENT_ENDED))


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
    for root in args.root:
        rows = outcomes(root).values()
        exhausted = sum(r.get('status') == 'request_budget_exhausted' for r in rows)
        stalled = sum(r.get('status') == 'session_stalled' for r in rows)
        missing = incomplete(root)
        print(f'{Path(root).name}: {exhausted} ended by the request budget and {stalled} by a stalled session '
              f'(both scored from their step traces); {len(missing)} incomplete for other reasons'
              + (': '+', '.join(missing) if missing else ''))
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


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k/n
    centre, half = p + z*z/(2*n), z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((centre-half)/(1+z*z/n), (centre+half)/(1+z*z/n))


def mcnemar(b, c):
    """Exact two-sided McNemar p-value from the discordant pair counts."""
    n = b + c
    if not n:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c)+1)) / 2**n
    return min(1.0, 2*tail)


def compare(args):
    """Paired comparison of two new arms (first attainment, the paper's metric; final state as well)."""
    pub = {(e['suite'], e['task'], e['case_id']): bool(e['success']) for e in published()}
    a_root, b_root = args.root
    a, b = outcomes(a_root), outcomes(b_root)
    na, nb = Path(a_root).name, Path(b_root).name
    tasks = sorted({k[1] for k in pub})
    print(f'| suite | task | published APPL | {na} | {nb} |')
    print('| --- | --- | --- | --- | --- |')
    for suite in SUITES:
        for task in tasks:
            keys = [k for k in pub if k[0] == suite and k[1] == task]
            if not keys:
                continue
            cell = lambda rows: f"{sum(bool(rows[k].get('first_attainment_success', rows[k]['success'])) for k in keys if k in rows)}/{sum(k in rows for k in keys)}"
            print(f"| {suite} | {task} | {sum(pub[k] for k in keys)}/{len(keys)} | {cell(a)} | {cell(b)} |")
    print()
    for metric, field in (('first attainment', 'first_attainment_success'), ('final state', 'success')):
        print(f'### {metric}')
        print(f'| suite | {na} | {nb} | paired cases | only {na} | only {nb} | exact McNemar p |')
        print('| --- | --- | --- | --- | --- | --- | --- |')
        for suite in (*SUITES, 'all'):
            keys = [k for k in pub if (suite == 'all' or k[0] == suite) and k in a and k in b]
            get = lambda rows, k: bool(rows[k].get(field, rows[k]['success'])) if field != 'success' else bool(rows[k]['success'])
            ka, kb = sum(get(a, k) for k in keys), sum(get(b, k) for k in keys)
            only_a = sum(get(a, k) and not get(b, k) for k in keys)
            only_b = sum(get(b, k) and not get(a, k) for k in keys)
            ia, ib = wilson(ka, len(keys)), wilson(kb, len(keys))
            print(f'| {suite} | {ka}/{len(keys)} [{ia[0]:.2f}, {ia[1]:.2f}] | {kb}/{len(keys)} [{ib[0]:.2f}, {ib[1]:.2f}] '
                  f'| {len(keys)} | {only_a} | {only_b} | {mcnemar(only_a, only_b):.3f} |')
        print()


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
    c = sub.add_parser('compare', help='paired comparison of exactly two roots')
    c.add_argument('--root', action='append', required=True)
    args = parser.parse_args()
    if args.command == 'compare' and len(args.root) != 2:
        parser.error('compare needs exactly two --root')
    {'run': run, 'summary': summary, 'compare': compare}[args.command](args)


if __name__ == '__main__':
    main()
