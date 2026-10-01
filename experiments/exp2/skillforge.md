# Exp2 with the SkillForge prompt system

SkillForge (github.com/agentics-robotics/skillforge) describes every skill by its priors and
makes those priors the interface through which an agent calls the skill. Its prompt system,
a catalog of 106 priors (`prompts/skillforge/PRIOR_CATALOG.md`) and a framework prompt
(`prompts/skillforge/skillforge.md`), is merged here into APPL's Exp2 construction and runtime
agents. This document defines the comparison between the published APPL prompts and the
merged system. It reports no result until both arms have been run and evaluated.

## What the SkillForge system changes

Every published stage prompt is kept unchanged; `--prompt-system skillforge` appends the
stage's section from `experiments/exp2/prompt_systems/skillforge/` and freezes the result in
the run's `inputs/`:

| Stage | Added |
| --- | --- |
| Cut | skills whose responsibility a prior can describe, with entry and exit readable from the state |
| Prior design | read the catalog (`read_catalog`; the index must be read before the prior plan is accepted); name catalog IDs per prior; grounding in applicability; physical meaning, unit and demonstrated range in parameter semantics; failure conditions from the catalog in limitations |
| Revision (h04) | return to the catalog for the observed failure; record contradicted priors as negative results |
| Final guidance | each policy's guidance as its prior card |
| Runtime agent | ground a policy's priors in the current state before calling; on failure decide whether the policy or the call failed |

`--prompt-system appl` freezes the published prompts byte-for-byte (revision and final
guidance included) and gives no catalog access. Everything else, the 12 demonstrations, the
preregistered skills and exit rules, training recipe, ID verification, cases and executor, is
shared.

## Running both arms through Codex

`--backend codex` runs every agent session through the local Codex login (`codex app-server`)
instead of `OPENAI_API_KEY` (`src/appl/codex.py`). The stage prompt replaces Codex's base
instructions; the session's APPL tools are its only tools; Codex's shell, files, browser,
plugins and other built-ins are disabled. Both arms use the same backend and model:

```bash
pixi run --locked python scripts/exp2_construct.py --root runs/exp2_gpt6sol/appl \
  --prompt-system appl --backend codex --model gpt-6-sol --gpu drawer_exchange=1 ...
pixi run --locked python scripts/exp2_construct.py --root runs/exp2_gpt6sol/skillforge \
  --prompt-system skillforge --backend codex --model gpt-6-sol --gpu drawer_exchange=1 ...
pixi run --locked python scripts/exp2_evaluate.py run --root runs/exp2_eval/appl \
  --construction-root runs/exp2_gpt6sol/appl --gpu 1 --gpu 2
pixi run --locked python scripts/exp2_evaluate.py summary --root runs/exp2_eval/appl \
  --root runs/exp2_eval/skillforge
```

The evaluation uses the 96 published APPL episodes (motion 40, task 40, composition 16) and
checks each initial state against its published pair.

## Differences from the paper's protocol

These apply equally to both arms, so they do not favor either prompt system; they do make
the published APPL numbers context rather than a baseline.

- Model: GPT-6 Sol (xhigh) instead of the paper's GPT-5.6 Sol (xhigh).
- Agent harness: Codex manages the conversation. The runtime agent's six-turn context
  projection is not applied, Codex may compact long sessions, and it may issue several tool
  calls in one response. Budgets count model requests, as in the paper.
- Tool arguments that violate a schema are returned to the agent as an error instead of
  being prevented by strict function calling.
- A turn that fails for a transient provider reason is continued after the Responses
  transport's bounded delays; `max_output_tokens` is left to Codex.
- Each arm runs its own Cut, so the two libraries can differ in segmentation as well as in
  priors.
