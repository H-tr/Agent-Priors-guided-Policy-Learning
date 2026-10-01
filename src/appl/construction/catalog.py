"""Read access to the SkillForge prior catalog for prior-design and revision sessions."""
from pathlib import Path
import re

from .journal_tools import schema

TOOL = schema('read_catalog', 'Read the prior catalog: an empty list returns its usage notes and index; '
              'catalog IDs (e.g. "A.2") return those entries in full.',
              dict(ids=dict(type='array', items=dict(type='string', pattern=r'^[A-E]\.[0-9]{1,2}$'), maxItems=8)))


def sections(path):
    text = Path(path).read_text()
    starts = [(m.start(), m.group(1)) for m in re.finditer(r'^### ([A-E]\.\d+)\b', text, re.M)]
    if not starts:
        raise ValueError('Prior catalog has no entries')
    entries = {ident: text[start:(starts[i+1][0] if i+1 < len(starts) else len(text))].rstrip()
               for i, (start, ident) in enumerate(starts)}
    return text[:starts[0][0]].rstrip(), entries


def read(path, ids):
    index, entries = sections(path)
    if not ids:
        return dict(content=index)
    unknown = [i for i in ids if i not in entries]
    if unknown:
        raise ValueError('Unknown catalog IDs: '+str(unknown))
    return dict(entries={i: entries[i] for i in ids})
