
HOW TO STRUCTURE THE WORK (SkillForge: priors as the interface)

Build the solution as **skills extracted by you, described by their priors, and called by your program during execution.**

A *prior* is any explicit structure you put into a policy: its representation (object or end-effector frames, keypoints, affordances, image-plane points, bounding volumes, relative poses), its objective and data (losses, auxiliary predictions such as future states or grasp poses, data augmentation, data generation), its architecture (equivariance, object-centric or relational structure, residual on a controller, hierarchy), its control or planning structure (motion planner, MPC, inverse kinematics, guarded moves, impedance, movement primitives, behavior tree, state machine, symbolic operators, programs), a physical model (pushing, friction, grasp stability, tool contact), or task and demonstration structure (phases, keyframes, invariances, relations, retries). Symbolic preconditions and effects are one kind of prior among many.

`ref/PRIOR_CATALOG.md` lists these kinds with how to implement them, what each tells a caller, when each helps or fails, how to test it, and literature. **Read it before designing skills, and consult it whenever a skill fails.**

## Skills

A skill is a closed-loop policy for one reusable piece of behavior, plus its **prior card**. The prior card is the skill's interface: it is how your program (and any later agent) decides whether, where and how to call the skill. Write it in `SKILLS.md` and keep a `CARD` dict with the same content in the skill's module:
- **priors:** the catalog entries it uses (by ID, e.g. `D.7`) and how each is instantiated here. New priors not in the catalog are welcome; describe them the same way.
- **grounding:** which objects, frames or quantities in the current state fill each role the priors need, and how the skill finds them from the observation.
- **parameters:** name, physical meaning, unit, and the range that was validated.
- **validity:** where the priors hold, with measured evidence (success on N fresh seeds, over which conditions and object counts), and where they break.
- **call:** how to invoke it and what it returns (success, failure or uncertain, with the measured effect).

Implement each skill as a step-wise controller that the per-step `get_action` can drive: for example `start(state, **params)`, then `act(state)` returning an action or `None` when finished, plus a `status` and a measured result. Put skills in `skills/` (a package with an `__init__.py`); the scorer accepts only the standard library, numpy, scipy and your own files.

## Extraction: choosing priors

For each skill, pick priors from the catalog that match what the skill must exploit, and build the policy in them: analytic, learned (small numpy models fitted to generated or demo data), or both. For the hardest skill, try at least two different priors or prior combinations and keep the one that measures better on fresh seeds. Record each tested prior, including negative results, in `PRIORS.md`: catalog ID, where the idea came from (demos, interaction, physics, literature), the experiment and the result with and without it.

## Execution: calling skills

approach.py is the caller. At each step it reads the state, grounds the candidate skills' priors in it, chooses a skill whose validity covers the situation, sets its parameters from the grounding, drives it, checks the measured effect, and recovers (retry with adjusted parameters, or another skill) when the effect is missing. When an episode fails, decide which layer failed: the skill (its prior did not hold, or its policy did not achieve the effect) or the caller (it chose, parameterized or ordered skills wrongly). Fix that layer, and record in the skill's card any change to its validity.

## Order of work

Your budget can end at any moment:
1. **Pin down the goal at every object count.** Confirm the exact success condition by experiment: what terminates an episode, and where each object must end up. The goal, including which object goes where, can change with the number of objects: check it at the smallest and the largest counts the environment accepts.
2. **Sketch candidate strategies as skill sequences,** with the priors each skill would use. Where the scene offers more than one way (for example a held object as a tool, carrying a container, or direct manipulation), run one short feasibility probe of each key effect and estimate the steps a full episode needs.
3. **Build the most promising skills and the caller until one full episode succeeds,** then commit.
4. **Improve the weakest skill or the caller,** judging every change on fresh seeds not used to develop it (at least eight, covering object counts).

## Choosing the final program

approach.py must always be the version with the highest measured full-episode success on fresh seeds across the object counts you can reach. A version with no full-episode success never replaces one that has some. Never route an object count to a skill you did not validate at that count. The evaluation step limit can be tight: aim to finish with at least a quarter of the limit to spare. Stop changing things when little budget remains. As a final check, build the policy exactly as the scorer does, `GeneratedApproach(action_space, observation_space, primitives={})` with `primitives` passed by keyword, then run `reset(state, info)` and `get_action(state)` for a full episode.

## Demonstrations (only if `demos/` exists)

Use them to find the skills and their priors: split them into phases and keyframes, see which object moves in each phase and in which frame the motion is simple, and read off parameter ranges. Treat the demonstrated strategy as one candidate; check demo-derived skills on fresh seeds.

## Skill library (only if `library/` exists)

`library/` holds skills that agents extracted for other tasks, with their prior cards. Read `library/README.md` first. A skill whose priors and grounding fit this task can be reused or adapted; re-verify it here, and record in its card what changed and why.

Keep SKILLS.md and PRIORS.md short and factual. approach.py's measured success comes first; keep it your best validated version at all times and commit it.
