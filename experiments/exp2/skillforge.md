# Exp2 with the SkillForge prompt system

SkillForge (github.com/agentics-robotics/skillforge) describes every skill by its priors and
makes those priors the interface through which an agent calls the skill. Its prompt system,
a catalog of 106 priors (`prompts/skillforge/PRIOR_CATALOG.md`) and a framework prompt
(`prompts/skillforge/skillforge.md`), is merged here into APPL's Exp2 construction and runtime
agents. This document defines the comparison between the published APPL prompts and the
merged system and reports its result (run 2026-10-01/02).

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

## Result (2026-10-02)

Both arms were constructed end to end (Cut, 3 priors per skill, training, ID verification,
one refined prior, final guidance; 17 skills, 68 policies each) and evaluated on all 96
published APPL episodes, through Codex with GPT-6 Sol (xhigh). Success is the paper's
metric, first goal attainment; final-state success gave identical counts.

| | motion | task | composition | all |
| --- | --- | --- | --- | --- |
| APPL prompts (rerun) | **30/40** | **34/40** | **6/16** | **70/96** (0.73, 95% CI 0.63–0.81) |
| SkillForge prompts | 23/40 | 30/40 | 5/16 | 58/96 (0.60, 95% CI 0.50–0.70) |
| published APPL (GPT-5.6 Sol, API; reference only) | 20/40 | 37/40 | 8/16 | 65/96 |

Paired over the same 96 cases, the APPL prompts alone succeed in 23 and the SkillForge
prompts alone in 11 (exact McNemar p = 0.058; motion 13 vs 6, p = 0.17; task 7 vs 3, p = 0.34;
composition 3 vs 2, p = 1.0). **The SkillForge prompt system did not outperform the APPL
prompts; it scored lower, a difference that is not significant at the 0.05 level.**

Per task (first attainment; published APPL in parentheses):

| task | motion APPL / SkillForge | task APPL / SkillForge | composition APPL / SkillForge |
| --- | --- | --- | --- |
| drawer_exchange | 6/8 / 1/8 (3/8) | 8/8 / 5/8 (7/8) | 0/4 / 2/4 (2/4) |
| buffer_swap | 5/8 / 7/8 (2/8) | 7/8 / 4/8 (8/8) | 1/3 / 1/3 (1/3) |
| constrained_retrieve_store | 6/8 / 6/8 (7/8) | 8/8 / 8/8 (8/8) | 3/3 / 1/3 (3/3) |
| covered_peg_assembly | 6/8 / 7/8 (1/8) | 8/8 / 7/8 (7/8) | 1/3 / 1/3 (0/3) |
| granular_pour_return | 7/8 / 2/8 (7/8) | 3/8 / 6/8 (7/8) | 1/3 / 0/3 (2/3) |

The gap comes mainly from the policies each arm built, not from the runtime agent. ID
verification (12 demonstration entry states per policy) shows the SkillForge libraries
weaker on drawer_exchange (red_transfer best 9/12 vs 12/12, mean 6.0 vs 9.2; blue_insert
8/12 vs 12/12, mean 4.5 vs 9.5) and granular grasp_lift_stage (7/12 vs 9/12, mean 3.5 vs 6.2),
exactly the tasks whose motion episodes it lost (1/8 vs 6/8 and 2/8 vs 7/8; in the granular
failures the runtime agent tried every grasp variant, none lifted the container, and it finished). It built stronger libraries for
constrained retrieve_tunnel_block (11/12 vs 7/12), close_drawer and buffer place_red, and
won buffer_swap and covered_peg motion. Mean best-policy ID success: 0.93 (APPL) vs 0.89.

How the system was used: in the SkillForge arm every one of the 34 prior plans cites
catalog entries (27 distinct IDs; most often A.2 object-frame actions, 32 plans; A.3, C.6,
E.7, D.1), from 134 `read_catalog` calls opening 368 entries. Construction took 56 agent
sessions per arm (1,547 vs 1,704 model requests). Total Codex use was 10% of the weekly plan.

### Limitations

- One construction per arm and task: a single weaker skill library decides a task's motion
  result, so library variance is confounded with the prompt system. Repeating construction
  (different seeds/sessions) would be needed to separate them; this was not done.
- Model and harness differ from the paper in both arms (see above); the published APPL row
  is not a baseline.
- Episodes whose agent session ended without a result were scored from their step traces
  (APPL: 2 request-budget, 5 stalled; SkillForge: 1 and 3). In every stall the agent had
  reported completion after its last action.
- Infrastructure incidents were resolved symmetrically and logged in
  `runs/exp2_gpt6sol_resets/resets.jsonl`: constrained_retrieve_store was rebuilt on another
  GPU after the first one filled (both arms); five design sessions killed by a disk-quota
  failure were restarted (3 APPL, 2 SkillForge); one SkillForge session was restarted under
  the one-restart rule; 72 verification demos interrupted by process stops were re-run;
  drawer_exchange and buffer_swap were moved to GPUs 5–6 (recorded in their provenance).

Per-episode outcomes: `results/exp2_skillforge/episodes.jsonl`; tables:
`results/exp2_skillforge/comparison.md`. Run directories (not versioned, about 50 GB) are
under `runs/exp2_gpt6sol/` and `runs/exp2_eval_gpt6sol/`.
