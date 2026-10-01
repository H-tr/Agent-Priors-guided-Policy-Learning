"""Move a construction run to another GPU, recorded as an operational amendment.

The GPU index only selects the device for training, checks and ID verification; it is not
part of any training request, verification identity or scientific input. The amendment is
appended to the run's provenance with the old and new index, the reason, and the
configuration hashes before and after, so `load()` continues to verify the configuration.

Usage (locked Exp2 environment):
  python scripts/exp2_reassign_gpu.py --run runs/exp2_gpt6sol/appl/drawer_exchange --gpu 5 --reason "..."
"""
import argparse
from pathlib import Path
import time

from appl.io import atomic, digest, read


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--reason', required=True)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    config, provenance = run/'configuration.json', run/'provenance.json'
    record = read(provenance)
    before = digest(config)
    if before != record['configuration_sha256']:
        raise ValueError('Configuration does not match its provenance; refusing to amend')
    cfg = read(config)
    old = cfg['gpu']
    if old == args.gpu:
        print(f'{run.name}: already on GPU {old}')
        return
    cfg['gpu'] = args.gpu
    atomic(config, cfg)
    record.setdefault('operational_amendments', []).append(dict(
        time=time.strftime('%Y-%m-%dT%H:%M:%S%z'), field='gpu', old=old, new=args.gpu, reason=args.reason,
        previous_configuration_sha256=before, configuration_sha256=digest(config)))
    record['configuration_sha256'] = digest(config)
    atomic(provenance, record)
    print(f'{run.name}: GPU {old} -> {args.gpu}')


if __name__ == '__main__':
    main()
