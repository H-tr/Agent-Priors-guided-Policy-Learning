# SkillForge prompt system (imported)

Copied unchanged from github.com/agentics-robotics/skillforge at `07a1153`:

- `PRIOR_CATALOG.md`: 106 priors in five families (A representation, B objective and data,
  C architecture, D control and planning, E task and demonstration structure), each with
  assumptions, implementation, what it tells a caller, when it helps or fails, how to test
  it, and checked references.
- `skillforge.md`: the SkillForge framework prompt (v6) for a single coding agent on KinDER:
  skills with prior cards, the program as caller, order of work.

The Exp2 construction and runtime agents use them through
`experiments/exp2/prompt_systems/skillforge/`, which appends a SkillForge section to each
unchanged APPL stage prompt and exposes the catalog through `read_public`.
