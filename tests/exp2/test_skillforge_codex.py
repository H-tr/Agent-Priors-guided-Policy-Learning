"""Offline checks for the SkillForge prompt system and the Codex backend; no model, API or GPU."""
import re
from types import SimpleNamespace

import pytest

from appl.io import ROOT

CATALOG = ROOT/'prompts/skillforge/PRIOR_CATALOG.md'


def test_catalog_index_and_entries():
    from appl.construction import catalog
    index = catalog.read(CATALOG, [])['content']
    assert '## Index' in index and not re.search(r'^### [A-E]\.\d+', index, re.M)
    entries = catalog.read(CATALOG, ['A.2', 'E.5'])['entries']
    assert entries['A.2'].startswith('### A.2 ') and entries['E.5'].startswith('### E.5 ')
    assert len(catalog.sections(CATALOG)[1]) == 106
    with pytest.raises(ValueError):
        catalog.read(CATALOG, ['Z.1'])


def test_skillforge_addenda_cover_every_stage():
    folder = ROOT/'experiments/exp2/prompt_systems/skillforge'
    assert {p.stem for p in folder.glob('*.md')} == {'cut', 'train', 'revision', 'final', 'hl'}
    # The revision template is formatted with the skill id; an addendum must not add fields.
    assert '{' not in (folder/'revision.md').read_text()


def test_tool_exception_becomes_a_diagnostic(tmp_path):
    from appl.construction.journal_tools import DesignTools, schema
    from appl.journal import Journal

    class Tools(DesignTools):
        def __init__(self):
            self.j = Journal(tmp_path/'j')
            self.budget = SimpleNamespace(max_tool_calls=8)

        def schemas(self):
            return [schema('read_file', 'Read a file.', dict(path=dict(type='string')))]

        def done(self):
            return False

        def dispatch(self, name, args):
            return dict(content=(tmp_path/args['path']).read_text())

    result = Tools().execute(dict(name='read_file', call_id='c1', arguments='{"path": "missing.py"}'))
    assert 'No such file' in result['error'] and result['repair_allowed']


def test_codex_tool_output_items():
    from appl.codex import _content
    assert _content({'a': 1}) == [dict(type='inputText', text='{"a":1}')]
    items = _content([dict(type='input_text', text='frame'),
                      dict(type='input_image', image_url='data:image/png;base64,AA', detail='high')])
    assert items == [dict(type='inputText', text='frame'), dict(type='inputImage', imageUrl='data:image/png;base64,AA')]


def test_codex_backend_is_selected_by_configuration(tmp_path, monkeypatch):
    from appl import api
    monkeypatch.setattr('subprocess.run', lambda *a, **k: SimpleNamespace(stdout='codex-cli test\n'))
    client = api.client(dict(api=dict(backend='codex', model='gpt-6-sol', reasoning_effort='xhigh')), tmp_path)
    assert client.codex and client.configuration()['model'] == 'gpt-6-sol'
    assert isinstance(api.client(dict(api=dict(model='gpt-5.6-sol')), tmp_path), api.ResponsesClient)
