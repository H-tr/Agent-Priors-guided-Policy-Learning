# Prior catalog

A prior is any explicit structure you put into a policy: what it observes and acts in
(A), what it is trained on and to predict (B), how the model is built (C), the
algorithmic, symbolic or physical structure it runs (D), and the structure of the
task and demonstrations it exploits (E). Symbolic preconditions and effects (D.8,
D.9, E.9) are one kind of prior among about a hundred here.

**How to use this file**
1. Read the index below in full before designing skills. Open the detailed entries
   of the priors you consider (search for `### <ID>`).
2. For each skill, choose the priors that match what it must exploit. Priors combine:
   a skill typically has a representation (A), a control or planning structure (D),
   often task structure from demos (E), and, if it learns, an objective and data
   recipe (B) and a model structure (C).
3. Build the skill's policy in those priors, and describe the skill through them in
   its prior card: IDs used and how each is instantiated, grounding in the scene,
   parameters, and the measured region where the priors hold.
4. Test a prior the way its entry says ("How to test it"): the same skill with and
   without it (the "without" arm is also a prior; name it), on fresh seeds.
5. When a skill fails, return here: the failure often matches an entry's "Fails when",
   and another entry is the fix.

**Your sandbox:** Python with numpy and scipy only; fully observed object-centric
state (no images); a simulator through `reset` and `step` only (no `set_state`), so
a rollout from the current state means reset plus replaying the executed actions,
and fast inner-loop planning usually runs on a numpy model you fit or derive. The
coding agent (you) plays the role of any LLM/VLM component at authoring time.
"Numpy-only feasible" in each entry says what survives these constraints.

Entries were compiled from a literature review (September 2026); every reference
was checked against arXiv, Crossref, proceedings pages or publisher sites.

## Index

**A. Representation and perception**

| ID | Prior | Numpy | Assumes |
| --- | --- | --- | --- |
| A.1 | Object-centric state | yes | The task-relevant state factors into a few discrete objects, each with a pose and a few attributes. |
| A.2 | Relative / object-frame actions (task-parameterized frames) | yes | The solving motion is invariant when expressed in the frame of the object being acted on, or as a relative pose between two objects. |
| A.3 | End-effector-centric frames and relative (delta) actions | yes | The next action depends mainly on local geometry around the gripper. |
| A.4 | Pose and rotation parameterization | yes | The coordinates of the action decide whether regression, averaging and interpolation across demos are well-posed. |
| A.5 | Keypoints (semantic / task keypoints) | yes | A few named 3D points on objects (handle, rim, tip, hole center) plus geometric costs between them specify the task. |
| A.6 | Affordance maps | partly | Where to interact, and in which direction, is a local property of part geometry. |
| A.7 | 2D image-plane points and pixel/plane-space actions | partly | Planar, quasi-static tasks (top-down pick, place or push) are well parameterized by a location in the same 2D grid as the observation. |
| A.8 | Bounding boxes and segmentation masks | partly | An object's extent (box) or support (mask) is enough to localize and isolate the task-relevant object. |
| A.9 | 3D point clouds and voxels | partly | Actions live in 3D and are best predicted in the same metric 3D frame as the observation. |
| A.10 | Scene graphs and relational state | yes | Task state, goals and success are captured by relations between entities (on, in, touching, grasped, left-of) that are invariant to exact poses. |
| A.11 | Optical flow and point tracks (object motion as the interface) | yes | A skill is captured by how points on the manipulated object should move, independent of the embodiment producing that motion. |
| A.12 | Spatial softmax / feature points | no | Task-relevant visual information reduces to a few 2D locations. |
| A.13 | Visual cue overlays and visual prompting | partly | Policies and VLMs use spatial relations better when those relations are rendered explicitly into the input rather than left implicit. |
| A.14 | Contact and wrench representations | partly | In contact phases, the relevant state is the contact state and the interaction wrench, not pose alone. |
| A.15 | Tactile representations | no | Fingertip contact geometry and slip carry information vision lacks: occluded contact, grasp stability and in-hand pose. |
| A.16 | Language and semantic features | partly | Task-relevant variables can be named (object classes, parts, relations, attributes). |
| A.17 | Dense descriptors and correspondence fields | partly | Points on different instances and poses of a category can be put in correspondence. |
| A.18 | Articulation and kinematic-structure representations | yes | Articulated objects move along low-dimensional joints (revolute, prismatic, screw). |

**B. Learning objective and data**

| ID | Prior | Numpy | Assumes |
| --- | --- | --- | --- |
| B.1 | Unimodal regression behavior cloning (MSE / L1) | yes | In each state the demonstrator's action is one value plus zero-mean noise. |
| B.2 | Multimodal explicit behavior cloning (mixture density, discretized / clustered actions) | yes | `p(a\|s)` has several distinct modes (grasp from the left or the right, different subtask orders, detour direction). |
| B.3 | Diffusion and flow-matching action objectives | partly | The distribution of actions or action chunks is complex and multimodal but smooth. |
| B.4 | Energy-based / implicit behavior cloning | yes | A scalar compatibility `E(s, a)` whose minimum is the expert action is easier to learn than a direct map `s → a`. |
| B.5 | Action chunking and temporal ensembling | yes | Expert actions are temporally correlated and partly non-Markovian (pauses, hesitations). |
| B.6 | Contrastive, time-contrastive and progress/value objectives | yes | A useful representation places states closer together as they get closer in time or closer to the goal. |
| B.7 | Dynamics-prediction auxiliaries (forward / future-state / video, inverse dynamics) | yes | Features that predict how the world changes under an action (forward), or which action explains a transition (inverse), are the features control needs. |
| B.8 | Target, keyframe and subgoal prediction (grasp pose, contact point, goal pose, waypoints) | yes | Manipulation is decided by a few sparse targets: where to grasp, where to place, where to make contact, and the next bottleneck pose. |
| B.9 | Reward shaping (potential-based, staged, LLM-written reward code) | yes | The designer knows the task's intermediate structure (reach → grasp → lift → place) and can encode it as a dense reward. |
| B.10 | Goal-conditioned hindsight relabeling (HER, GCSL, play data) | yes | Any trajectory is an optimal demonstration of reaching the states it actually reached, so a failed attempt at goal g is a success for some other goal g'. |
| B.11 | Learned rewards from demonstrations (IRL, adversarial imitation, SQIL) | partly | The demonstrator is near-optimal for an unknown reward. |
| B.12 | Success classifiers, VLM success detectors and programmatic success predicates | yes | Success is easier to recognize than to specify. |
| B.13 | Observation and geometric data augmentation | yes | The correct action is invariant or equivariant to known nuisance transformations: pixel shifts, color, small state noise, rigid transforms of the whole scene, or changes… |
| B.14 | Noise injection and corrective trajectory augmentation | yes | Policy errors push the robot off the demo manifold. |
| B.15 | Domain randomization | yes | The test condition is one draw from a distribution over physical and scene parameters. |
| B.16 | Demo-based data generation (MimicGen family) | yes | A demo splits into object-centric subtask segments. |
| B.17 | Synthetic expert data from planners, trajectory optimization and privileged / scripted experts | yes | In simulation, an expert with privileged information (full state, model, simulator resets) or unlimited compute can solve each instance. |
| B.18 | Interactive correction (DAgger, HG-DAgger) | yes | An expert can label the states the learner itself visits. |
| B.19 | Self-supervised data collection and success-filtered self-improvement | yes | The robot can produce its own training data by trying things, given an automatic outcome label. |
| B.20 | Curriculum (reverse / start-from-demo-states, jump-start, automatic) | yes | The task becomes learnable if training starts from easy instances (near the goal, small perturbations, simple parameters) and difficulty grows with competence. |
| B.21 | Residual learning on a base policy | yes | A base controller (scripted, planner-based or BC) is roughly right, and what remains is a small, learnable correction. |
| B.22 | Regularization toward a prior / reference policy (BC terms, KL, demo replay) | yes | A reference policy (BC from demos, or a previously learned skill) encodes good behavior. |

**C. Architecture and model structure**

| ID | Prior | Numpy | Assumes |
| --- | --- | --- | --- |
| C.1 | Group-equivariant policies (SO(2) / SE(2) / SE(3) / SIM(3)) | yes | Moving the whole scene by a group element g (rotation, translation, possibly scale) should move the correct action by the same g, so π(g·s) = g·π(s). |
| C.2 | Object-frame descriptors and task-frame selection (Neural Descriptor Fields, task-parameterized models) | yes | A skill is defined by a relative pose between the end effector (or a held object) and a frame attached to a task-relevant object. |
| C.3 | Permutation-invariant set encoders over objects (Deep Sets, Set Transformer, PointNet) | yes | The scene is an unordered set of entities. |
| C.4 | Relational / graph-structured policies (graph networks, relational RL) | yes | What matters is the pairwise relations between entities (contact, on-top-of, distance, alignment), and the same relation function applies to every pair. |
| C.5 | Object slots and object-token attention (slot attention, object-centric transformers) | yes | A scene is best represented as a fixed-width token per object. |
| C.6 | Diffusion policies | partly | The demo action distribution is multimodal and high-dimensional (whole action sequences). |
| C.7 | Flow-matching policies | partly | Same multimodality prior as diffusion, but the policy is a learned velocity field that transports noise to actions along near-straight paths. |
| C.8 | Action chunking with a latent style variable (ACT / CVAE) | yes | Predicting k future actions at once lowers the effective horizon and reduces compounding error. |
| C.9 | Mixture-density and discretized multimodal action heads (MDN, GMR, BeT, VQ-BeT) | yes | p(a\|s) is a small number of modes, each roughly Gaussian or a discrete bin plus a small continuous offset. |
| C.10 | Energy-based / implicit policies | yes | A good action is easier to recognize than to generate. |
| C.11 | Hierarchical policies and options | yes | Behaviour decomposes in time into reusable sub-behaviours, each with its own start condition and stopping condition. |
| C.12 | Latent skill spaces and skill-conditioned policies | yes | Useful behaviours lie on a low-dimensional manifold of trajectory segments. |
| C.13 | Mixture of experts and modular policies | yes | The state space splits into regimes (contact modes, task variants, robot bodies), each simple enough for its own expert. |
| C.14 | Residual policies on top of a base controller | yes | An existing controller (scripted, model-based, or a BC policy) is mostly right. |
| C.15 | Controller-structured action spaces (impedance / operational-space outputs, movement-primitive layers) | yes | The right thing to learn is the parameters of a well-understood controller (setpoints, stiffness, damping, attractor goals), not raw torques. |
| C.16 | Optimization as the policy (differentiable MPC, trajectory-optimization and planning layers) | yes | Good behaviour is the solution of a short-horizon optimal-control problem. |
| C.17 | Pretrained perception representations (R3M, MVP, VC-1, DINO-style features) | no | Visual features pretrained on large human-video or image corpora (time-contrastive, masked autoencoding, self-distillation) already encode what matters for manipulation. |
| C.18 | Vision-language-action models (VLAs) as the policy backbone | no | A vision-language model pretrained on web data and then co-trained or fine-tuned on large multi-robot datasets carries semantic and physical priors. |
| C.19 | LLM/VLM planners and code generation as architecture components | yes | A language model's commonsense and coding ability can do the high-level decomposition, parameter choice and constraint specification, provided each proposal is grounded… |
| C.20 | Learned world models and latent dynamics (PETS, Dreamer, TD-MPC, structured world models) | partly | The environment's dynamics can be learned from interaction and then used to plan or to train a policy in imagination, which is more sample-efficient than model-free lear… |
| C.21 | Physics-structured dynamics models (interaction networks, Lagrangian/Hamiltonian networks, sparse physical regression) | yes | The dynamics obey known physical structure: energy conservation or Euler–Lagrange form, pairwise object interactions, or sparsity in a small library of physical terms. |
| C.22 | Spatially grounded architectures (spatial-softmax keypoints, dense action maps, attention over 3D points) | yes | Task-relevant information is where things are, and actions are best expressed as locations or poses in the same space as the observation. |

**D. Control, planning and structured policies**

| ID | Prior | Numpy | Assumes |
| --- | --- | --- | --- |
| D.1 | Inverse kinematics (damped least squares / multi-start) | yes | Goals are best stated in task space (end-effector pose, keypoint position), and a kinematic map x = f(q) is known or locally linearizable. |
| D.2 | Sampling-based motion planning (RRT, RRT-Connect, PRM, RRT*) | yes | The task reduces to collision-free transit between known start and goal configurations. |
| D.3 | Trajectory optimization (CHOMP, TrajOpt, KOMO) | yes | A good motion is a local optimum of smoothness plus constraint costs (goal, clearance, joint limits, orientation), and it can be reached from a sensible initial guess su… |
| D.4 | Sampling-based MPC (MPPI, CEM / iCEM, predictive sampling) | partly | A forward model (simulator, analytic or learned) predicts the outcomes of short action sequences well enough, and the task can be written as a cost. |
| D.5 | Gradient-based optimal control (iLQR / DDP) | partly | Dynamics are locally smooth and the cost is locally quadratic. |
| D.6 | Real2sim, system identification and digital twins | yes | The environment follows a parametric physics model. |
| D.7 | Contact-implicit optimization and differentiable/smoothed physics | partly | The contact schedule (which face, when, sticking or sliding) need not be fixed in advance. |
| D.8 | Task and motion planning (PDDLStream, logic-geometric programming) | partly | Long-horizon tasks factor into a discrete plan of symbolic operators (preconditions and effects) whose continuous parameters (grasps, placements, contact points, paths)… |
| D.9 | Learned symbolic operators, predicate invention and bilevel planning | yes | Demos and transitions can be abstracted into a few relational predicates over objects such that each skill has compact lifted STRIPS-style effects. |
| D.10 | LLM/VLM task planners with affordance grounding (SayCan, Inner Monologue) | partly | Common-sense sequencing from a language model is useful only when it is grounded by per-skill feasibility (affordance/value) estimates and closed-loop feedback: success… |
| D.11 | Behavior trees | yes | Task logic is a modular tree of conditions and actions, re-evaluated every tick. |
| D.12 | Finite-state machines and hybrid automata (mode switching) | yes | Behavior consists of discrete modes (approach, contact, push-sticking, push-sliding, retreat), each with its own continuous controller or dynamics. |
| D.13 | Funnels and sequential composition (preconditions as regions of attraction) | yes | Each skill is a feedback controller with a domain (region of attraction / preimage) and a goal set. |
| D.14 | Dynamic Movement Primitives (DMPs) | yes | A motion is a stable second-order attractor to a goal g, plus a learned forcing term that depends on a phase variable and decays to zero. |
| D.15 | Probabilistic and via-point movement primitives (ProMP, VMP, ProDMP) | yes | Demos are samples from a Gaussian distribution over basis-function weights. |
| D.16 | Waypoint / keyframe primitives | yes | A demo is captured by a few key poses (at gripper open/close events, velocity near zero, or contact changes), with a simple interpolator or planner between them. |
| D.17 | Parameterized action primitives (RAPS, MAPLE, PRIME) | yes | The action space is better expressed as a small library of primitives (reach, grasp, push, lift, gripper open/close), each with a few continuous parameters. |
| D.18 | Impedance, admittance and operational-space control | partly | In contact, the controller should regulate the dynamic relation between motion and force, i.e., a virtual spring-damper F = K(x_d − x) + D(ẋ_d − ẋ), rather than position… |
| D.19 | Hybrid force/position control, task-frame formalism and guarded moves | yes | In a task frame, each axis is either position-controlled or force-controlled according to natural constraints (you cannot move into a surface, and you cannot push along… |
| D.20 | Quasi-static pushing mechanics (limit surface, motion cones) | yes | Planar pushing is quasi-static: inertia is negligible, and the support friction's limit surface (often approximated as an ellipsoid) maps the applied wrench to object tw… |
| D.21 | Grasp analysis (friction cones, form/force closure, quality metrics) | yes | Grasp stability follows from contact geometry and Coulomb friction. |
| D.22 | Keypoint / constraint-based policies (kPAM, KETO, VoxPoser, ReKep; tool use) | yes | Task success can be stated as geometric costs or constraints on a few semantic keypoints on objects, tools and the gripper, e.g., "tool tip within 1 cm of the target, ha… |
| D.23 | Code-as-policies and verified skill libraries (CaP, ProgPrompt, Voyager, Demo2Code) | yes | A policy is best expressed as a readable program that composes state queries and control primitives with ordinary control flow and arithmetic. |
| D.24 | Programmatic and neuro-symbolic policies (program synthesis, NS-CL, neuro-symbolic predicates) | yes | A policy is a program in a small DSL. |

**E. Task and demonstration structure**

| ID | Prior | Numpy | Assumes |
| --- | --- | --- | --- |
| E.1 | Waypoint / keyframe extraction | yes | A demo is fully described by a short sequence of keyframes (waypoints) joined by a simple interpolator or controller. |
| E.2 | Changepoint segmentation into phases and skills | yes | Demos are concatenations of a few recurring sub-behaviours. |
| E.3 | Bottleneck states and option discovery | yes | Good subgoals are states that many successful trajectories pass through (bottlenecks, high betweenness) or that separate weakly connected regions of the state space. |
| E.4 | Object-relative reference frames and demo transformation | yes | Within each phase the motion is fixed relative to one object's frame (the grasped object or the target). |
| E.5 | Task symmetries and invariances (rigid motion, permutation, object count, time-scaling) | yes | The outcome is unchanged (or changes predictably) under global translation or planar rotation (SE(2)/SO(3) equivariance), permutation of identical objects, added irrelev… |
| E.6 | Goal-conditioned structure and hindsight relabeling | yes | A skill is a family π(a \| s, g) indexed by a goal g. |
| E.7 | Contact-mode sequences and hybrid dynamics | yes | Manipulation is a hybrid system. |
| E.8 | Spatial relations and scene graphs as goals | yes | Task goals are sets of relations between objects (on, in, left-of, aligned-with, near), invariant to exact poses. |
| E.9 | Symbolic operators: learned preconditions and effects | yes | Each skill behaves like a STRIPS/PDDL operator. |
| E.10 | Affordances and cross-embodiment retargeting | partly | Where and how to interact (contact region + post-contact motion direction) belongs to the object and is shared across agents (human → robot, robot A → robot B). |
| E.11 | Tool-use structure (grasp part vs. functional part) | yes | A tool splits into a grasp part and a functional part (effector point + effect direction). |
| E.12 | Corrective data against compounding error (DAgger family) | yes | A policy fitted only to expert states drifts into unseen states. |
| E.13 | Runtime error detection, retry and fallback structure | yes | Failures can be detected at runtime from state (violated precondition, stalled progress, unexpected effect). |
| E.14 | Skill chaining and reusable skill libraries | yes | Complex tasks are compositions of reusable skills, each with an explicit initiation set and termination set. |
| E.15 | Reward inference from demonstrations and preferences (IRL) | partly | Demonstrators (near-)optimise an unknown reward that is simple in the right features. |
| E.16 | Constraint inference from demonstrations | yes | Demos satisfy hard constraints (keep upright, stay above the table, avoid region R, keep A aligned with B's axis). |
| E.17 | Temporal task specifications (LTL / reward machines) | yes | Task success is a temporal pattern over predicates (eventually X, X before Y, always not Z) that a small automaton can represent. |
| E.18 | Language as a structuring prior | yes | Natural language supplies a vocabulary of reusable subtasks and parameters. |
| E.19 | Success detectors and self-verification | yes | Success is easier to recognise than to produce. |
| E.20 | Calibrated and conformal confidence of skills | yes | A skill's reliability should be reported as a calibrated probability, or as a distribution-free bound estimated on held-out episodes from the same distribution. |

---

# A. Representation and perception priors

What the policy observes and acts in. Each entry is a choice of coordinates, entities or features. Every ablation's "without" arm is also a representation choice (usually raw world-frame state), so name it when reporting. Sandbox reminder: agents have Python + numpy + scipy and fully observed object-centric state, with no images and no torch. "Numpy-only feasible" says what survives under that constraint.

Quick map for the numpy sandbox:
- **Directly usable:** A.1, A.2, A.3, A.4, A.5, A.10, A.11, A.18
- **Usable as a numeric analog:** A.6, A.7, A.8, A.9, A.13, A.14, A.16, A.17
- **Needs images, torch or sensors:** A.12, A.15

---

### A.1 Object-centric state
- **Assumes:** The task-relevant state factors into a few discrete objects, each with a pose and a few attributes. The action depends on those entities and not on the background or on untouched objects.
- **Executable form:** Build the observation as a set of per-object records `{id, class, p, R, size, v, grasped}`. Keep only task-relevant objects: those whose pose changes in demos, or that come within contact distance of the EE or a moved object. The policy is a function of the kept records, and permutation-invariant over interchangeable ones (sort by a canonical key, or pool).
- **Numpy-only feasible:** yes. This is the sandbox's native state. Relevance selection is simple arithmetic on demo trajectories (per-object displacement and minimum EE distance).
- **Describes the skill to a caller as:** object roles it binds (e.g. `obj: thing moved`, `target: receptacle`), the per-object fields it reads, and an explicit statement that all other objects are ignored.
- **Helps when:** there are distractors, few demos, or scenes composed of reusable parts.
- **Fails when:** the relevant state is not object-like (piles, fluids, cloth), or the key attribute is missing from the record (mass, friction, articulation state).
- **How to test it:** Fit the same policy class on (a) the full state vector and (b) task-relevant object records only. Evaluate success rate in fresh sim scenes with 0/2/5 added distractors and randomized untouched objects, and report the success drop per distractor.
- **References:**
  - Y. Zhu, A. Joshi, P. Stone, Y. Zhu. "VIOLA: Imitation Learning for Vision-Based Manipulation with Object Proposal Priors." CoRL 2022. arXiv:2210.11339
  - F. Locatello et al. "Object-Centric Learning with Slot Attention." NeurIPS 2020. arXiv:2006.15055
  - A. Chapin, E. Dellandrea, L. Chen. "Is an object-centric representation beneficial for robotic manipulation?" ROBOVIS 2025. arXiv:2506.19408

### A.2 Relative / object-frame actions (task-parameterized frames)
- **Assumes:** The solving motion is invariant when expressed in the frame of the object being acted on, or as a relative pose between two objects. World-frame variation across demos is nuisance.
- **Executable form:** For each demo phase, express the EE (or held-object) poses in every candidate frame: `T_frame^-1 · T_ee`. Select the frame with the lowest across-demo variance at the phase's end, or fuse several frames TP-GMM-style (product of Gaussians weighted by inverse covariance). At run time, `T_ee_target = T_frame(now) · T_rel`, and re-read `T_frame` every step. For placement, learn the cross-pose of the held object relative to the target object.
- **Numpy-only feasible:** yes. SE(3) composition uses `scipy.spatial.transform.Rotation`, and frame selection is per-frame variance.
- **Describes the skill to a caller as:** the reference frame(s) per phase (object/part name), the target relative pose, and a tolerance (covariance). It is valid while the reference object's pose is observed and the object is rigid.
- **Helps when:** object placements vary across scenes, and when extrapolating to layouts outside the demo range.
- **Fails when:** the motion is constrained by world-fixed structure (gravity, table edge, joint limits), when a symmetric object makes the frame ambiguous, or when the wrong frame is chosen from too few demos.
- **How to test it:** Use world-frame targets vs object-frame targets fitted from the same demos. Measure success rate and final relative-pose error in fresh scenes with object poses sampled (i) inside and (ii) outside the demo range.
- **References:**
  - S. Calinon. "A tutorial on task-parameterized movement learning and retrieval." Intelligent Service Robotics 9:1–29, 2016. doi:10.1007/s11370-015-0187-9
  - C. Pan, B. Okorn, H. Zhang, B. Eisner, D. Held. "TAX-Pose: Task-Specific Cross-Pose Estimation for Robot Manipulation." CoRL 2022. arXiv:2211.09325
  - W. Zhou, B. Jiang, F. Yang, C. Paxton, D. Held. "HACMan: Learning Hybrid Actor-Critic Maps for 6D Non-Prehensile Manipulation." CoRL 2023. arXiv:2305.03942 (object-centric action = contact point on object + motion parameters)
  - H. Yang et al. "Beyond World-Frame Action Heads: Motion-Centric Action Frames for Vision-Language-Action Models." arXiv:2605.11809, 2026.

### A.3 End-effector-centric frames and relative (delta) actions
- **Assumes:** The next action depends mainly on local geometry around the gripper. Observations and actions expressed in the EE frame, as deltas or as a trajectory relative to the EE pose at chunk start, remove dependence on absolute robot and world placement.
- **Executable form:** Observe objects as `T_ee^-1 · T_obj`. The action is a Δpose in the EE frame, or a chunk of poses relative to `T_ee(t0)` (UMI "relative trajectory"). Optionally drop absolute proprioception.
- **Numpy-only feasible:** yes.
- **Describes the skill to a caller as:** action frame = EE, inputs = object poses in the EE frame, output = Δpose chunk. It does not depend on base placement, and it states whether it needs absolute proprioception.
- **Helps when:** base, camera, table height or object placement shifts, or when transferring across embodiments.
- **Fails when:** the goal is an absolute world location or the task respects workspace limits. Integrating deltas open-loop also drifts.
- **How to test it:** Compare absolute world-frame observations and actions with EE-relative observations and actions from the same demos. Evaluate success rate in fresh scenes with shifted robot base, table height and object region.
- **References:**
  - K. Hsu, M. J. Kim, R. Rafailov, J. Wu, C. Finn. "Vision-Based Manipulators Need to Also See from Their Hands." ICLR 2022. arXiv:2203.12677
  - C. Chi, Z. Xu, C. Pan, et al. "Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots." RSS 2024. arXiv:2402.10329
  - J. Zhao et al. "Do You Need Proprioceptive States in Visuomotor Policies?" arXiv:2509.18644, 2025.

### A.4 Pose and rotation parameterization
- **Assumes:** The coordinates of the action decide whether regression, averaging and interpolation across demos are well-posed. Continuous rotation representations and position setpoints (rather than velocities) give smoother targets.
- **Executable form:**
  - Store orientations as 6D (the first two columns of R), with Gram–Schmidt back to R. Use rotvec only for small deltas.
  - Canonicalize the quaternion sign before averaging, and interpolate with slerp.
  - Never average Euler angles.
  - Output pose setpoints to a low-level controller instead of raw velocities.
  - For symmetric objects, canonicalize the rotation modulo the symmetry group before fitting.
- **Numpy-only feasible:** yes (`Rotation`, `Slerp`, `Rotation.mean`).
- **Describes the skill to a caller as:** action = target EE pose (position in m, rotation as 6D or rotvec) in frame F at rate f, with tolerances ε_pos and ε_rot, plus the declared symmetry of the object.
- **Helps when:** fitting or averaging orientations across demos, with large yaw ranges, or with multimodal grasp orientations.
- **Fails when:** the task needs velocity or force profiles (dynamic tasks), or an undeclared object symmetry makes targets multimodal.
- **How to test it:** Fit the same regressor on Euler vs 6D targets, and on unsigned vs sign-canonical quaternions. Measure goal orientation error and success in fresh scenes spanning the full yaw range. Separately, compare position vs velocity actions.
- **References:**
  - Y. Zhou, C. Barnes, J. Lu, J. Yang, H. Li. "On the Continuity of Rotation Representations in Neural Networks." CVPR 2019. arXiv:1812.07035
  - A. R. Geist, J. Frey, M. Zhobro, A. Levina, G. Martius. "Learning with 3D rotations, a hitchhiker's guide to SO(3)." ICML 2024. arXiv:2404.11735
  - C. Chi, Z. Xu, S. Feng, et al. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." RSS 2023 (IJRR 2025). arXiv:2303.04137 (position vs velocity action-space comparison)

### A.5 Keypoints (semantic / task keypoints)
- **Assumes:** A few named 3D points on objects (handle, rim, tip, hole center) plus geometric costs between them specify the task. Keypoints absorb intra-category shape variation that a 6-DoF template pose cannot.
- **Executable form:** Define K keypoints per object as offsets in the object frame. Get them from geometry (bbox top-center, face centers, principal axis ends) or from demos (the mean contact point in the object frame at grasp or release). Transform them to the world frame each step. Express each stage as costs or constraints on keypoints, e.g. `‖kp_tip − kp_hole‖ < ε` or axis alignment. Solve for the EE pose with `scipy.optimize.minimize`, or regress actions from keypoint coordinates (KAT).
- **Numpy-only feasible:** yes, when object pose and extents are in state. Keypoint offsets can be estimated from demo contacts.
- **Describes the skill to a caller as:** keypoints `{name: (object, local offset)}`, the per-stage constraint functions, and success = constraint residual below ε. It is valid when the object is rigid and the keypoint locations are correct for the instance.
- **Helps when:** instances differ in size or shape within a category, when you want checkable success, and when composing constraints across stages.
- **Fails when:** objects deform, too few points leave orientation under-determined (use ≥3 non-collinear points or an axis), or a mis-chosen keypoint silently gives a wrong goal.
- **How to test it:** Compare an object-pose target with a keypoint-constraint target. Evaluate in fresh scenes with object instances scaled or stretched by ±30%, measuring success rate and constraint residual at the end of the episode.
- **References:**
  - L. Manuelli, W. Gao, P. Florence, R. Tedrake. "kPAM: KeyPoint Affordances for Category-Level Robotic Manipulation." ISRR 2019. arXiv:1903.06684
  - Z. Qin, K. Fang, Y. Zhu, L. Fei-Fei, S. Savarese. "KETO: Learning Keypoint Representations for Tool Manipulation." ICRA 2020. arXiv:1910.11977
  - N. Di Palo, E. Johns. "Keypoint Action Tokens Enable In-Context Imitation Learning in Robotics." RSS 2024. arXiv:2403.19578
  - W. Huang, C. Wang, Y. Li, R. Zhang, L. Fei-Fei. "ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation." CoRL 2024. arXiv:2409.01652

### A.6 Affordance maps
- **Assumes:** Where to interact, and in which direction, is a local property of part geometry. It can be scored per point, transfers across instances, and is concentrated on a small region.
- **Executable form:** Score candidate interaction points (surface samples or pixels) with p(success | point, direction), then act at the argmax or sample from it. Learn the scores from interaction trials (Where2Act), from human video (VRB), or compose attraction and avoidance value maps and plan through them (VoxPoser).
- **Numpy-only feasible:** partly. Sample points on object primitives in the object frame. Score them with a KDE over demo contact points, updated by sim-trial outcomes. A 3D voxel value map is a numpy array searched with argmin or A*. Image-based learned affordance needs images/torch.
- **Describes the skill to a caller as:** interaction region on object X (object-frame points with scores), preferred approach or pull direction, and confidence. It is valid for parts similar to those seen.
- **Helps when:** acting on articulated objects (where to pull), using tools, meeting novel instances, or seeding RL exploration.
- **Fails when:** the affordance depends on hidden state (locked, heavy), the map changes after contact in multi-step interactions, or human-hand affordances don't fit the gripper.
- **How to test it:** Compare centroid or random contact points with the affordance argmax. Measure first-attempt success and the number of sim trials to first success on fresh object instances.
- **References:**
  - K. Mo, L. Guibas, M. Mukadam, A. Gupta, S. Tulsiani. "Where2Act: From Pixels to Actions for Articulated 3D Objects." ICCV 2021. arXiv:2101.02692
  - S. Bahl, R. Mendonca, L. Chen, U. Jain, D. Pathak. "Affordances from Human Videos as a Versatile Representation for Robotics" (VRB). CVPR 2023. arXiv:2304.08488
  - W. Huang, C. Wang, R. Zhang, Y. Li, J. Wu, L. Fei-Fei. "VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models." CoRL 2023. arXiv:2307.05973
  - Y. Tang, W. Huang, et al. "UAD: Unsupervised Affordance Distillation for Generalization in Robotic Manipulation." arXiv:2506.09284, 2025.

### A.7 2D image-plane points and pixel/plane-space actions
- **Assumes:** Planar, quasi-static tasks (top-down pick, place or push) are well parameterized by a location in the same 2D grid as the observation. Translation equivariance lets local evidence be reused everywhere, and a score map represents multimodal targets.
- **Executable form:** Discretize the image or table plane. Compute a pick score map, then a place map conditioned on the picked crop (Transporter cross-correlation). The action is `(u, v[, θ-bin])` at the argmax, lifted to 3D by depth or surface height. Named primitives (pick, place, push) execute below it. Hierarchical variants use a VLM to predict 2D points or a 2D path that a low-level policy follows (RoboPoint, HAMSTER).
- **Numpy-only feasible:** partly. Use a world x–y grid over the table and a score per cell computed from object state, e.g. a Gaussian around demo-derived offsets relative to the target object. Take the argmax, look up height, and use discrete θ bins. True pixel-space learning needs images/torch.
- **Describes the skill to a caller as:** action = `(x, y[, θ])` on plane P plus a primitive name. Height comes from the surface. It is valid for top-down, quasi-static tasks, and multiple peaks mean multiple valid targets.
- **Helps when:** doing pick-and-place, rearrangement, kitting or pushing, when several targets are valid, or when demos are few.
- **Fails when:** the task needs full 6-DoF approach directions, stacked or occluded targets, or dynamics.
- **How to test it:** Compare direct regression of `(x, y, θ)` with argmax over a grid score map. Measure success in fresh scenes with 2+ equally valid targets (regression averages them and misses) and with shifted layouts.
- **References:**
  - A. Zeng, P. Florence, J. Tompson, et al. "Transporter Networks: Rearranging the Visual World for Robotic Manipulation." CoRL 2020. arXiv:2010.14406
  - M. Shridhar, L. Manuelli, D. Fox. "CLIPort: What and Where Pathways for Robotic Manipulation." CoRL 2021. arXiv:2109.12098
  - W. Yuan, J. Duan, V. Blukis, et al. "RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics." CoRL 2024. arXiv:2406.10721
  - Y. Li, Y. Deng, J. Zhang, et al. "HAMSTER: Hierarchical Action Models For Open-World Robot Manipulation." ICLR 2025. arXiv:2502.05485

### A.8 Bounding boxes and segmentation masks
- **Assumes:** An object's extent (box) or support (mask) is enough to localize and isolate the task-relevant object. Masking out the rest removes spurious correlations.
- **Executable form:** Run an open-vocabulary detector (box), then SAM (mask), and produce per-object crops, masks or masked point clouds. Condition the policy on the target's box or mask (MOO) or on masked point clouds (GROOT). With state: an oriented bounding box `(center, R, half_extents)` per object.
- **Numpy-only feasible:** partly. OBBs come from pose + size, or from PCA of vertices if a mesh is given. Box-derived features include top-face center, support height, clearance, IoU or overlap, and point-in-box containment tests. Masks need images.
- **Describes the skill to a caller as:** the target selected by name or box, which box features are used (top center, extent), and a validity note ("object approximately box-like, not heavily occluded").
- **Helps when:** scenes are cluttered, or when placing, stacking or containment needs clearance math. Open-vocabulary detection brings new instances.
- **Fails when:** objects are concave or thin (mug handle, ring) and a box misrepresents them. Detection and segmentation errors also propagate with no recovery.
- **How to test it:** Compare full-scene input with target box or masked input in fresh scenes with distractors and novel instances. In numpy, compare centroid-only placement with box-aware placement (top-face and clearance) across objects of varied size, measuring success and collision rate.
- **References:**
  - A. Kirillov, E. Mintun, N. Ravi, et al. "Segment Anything." ICCV 2023. arXiv:2304.02643
  - S. Liu et al. "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection." ECCV 2024. arXiv:2303.05499
  - A. Stone, T. Xiao, Y. Lu, et al. "Open-World Object Manipulation using Pre-trained Vision-Language Models" (MOO). CoRL 2023. arXiv:2303.00905
  - Y. Zhu, Z. Jiang, P. Stone, Y. Zhu. "Learning Generalizable Manipulation Policies with Object-Centric 3D Representations" (GROOT). CoRL 2023. arXiv:2310.14386

### A.9 3D point clouds and voxels
- **Assumes:** Actions live in 3D and are best predicted in the same metric 3D frame as the observation. 3D structure gives viewpoint invariance and spatial locality.
- **Executable form:** Fuse RGB-D into a workspace-frame point cloud and crop it. Then either:
  - encode it and predict the next keyframe as the argmax over voxels or points plus a rotation bin and gripper state (PerAct, Act3D), or
  - condition a continuous policy on a compact point-cloud embedding (DP3).

  RVT re-renders virtual views instead of voxelizing.
- **Numpy-only feasible:** partly. Sample surface points from object primitives or meshes, and voxelize with `np.floor`. Use `scipy.spatial.cKDTree` for nearest neighbors, clearance and collision checks, and Kabsch/ICP for alignment. A hand-designed next-best-voxel is the argmax of a score over the grid. Learned point encoders need torch.
- **Describes the skill to a caller as:** keyframe target = voxel or point in the workspace frame + rotation bin + gripper open/close. It also reports workspace bounds and resolution, and is valid when the geometry is sensed.
- **Helps when:** doing 6-DoF keyframe tasks, handling viewpoint changes, or doing collision-aware placement with few demos.
- **Fails when:** voxel size is coarser than the tolerance, depth fails on transparent or shiny objects, compute is limited, or deformable detail matters.
- **How to test it:** Compare object-pose-only planning with pose + sampled geometry points for collision-aware placement. Also sweep voxel resolution. Measure success and placement error in fresh cluttered scenes.
- **References:**
  - M. Shridhar, L. Manuelli, D. Fox. "Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation." CoRL 2022. arXiv:2209.05451
  - T. Gervet, Z. Xian, N. Gkanatsios, K. Fragkiadaki. "Act3D: 3D Feature Field Transformers for Multi-Task Robotic Manipulation." CoRL 2023. arXiv:2306.17817
  - Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, H. Xu. "3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations." RSS 2024. arXiv:2403.03954
  - A. Goyal et al. "RVT: Robotic View Transformer for 3D Object Manipulation." CoRL 2023. arXiv:2306.14896

### A.10 Scene graphs and relational state
- **Assumes:** Task state, goals and success are captured by relations between entities (on, in, touching, grasped, left-of) that are invariant to exact poses.
- **Executable form:** Nodes are objects, the gripper and support surfaces, with attributes. Edges are predicates computed by geometric tests on poses and extents. Skill preconditions, effects and success are predicates on the graph. Demos are segmented where edges change, and each segment becomes a skill whose goal is the edge change (ORION: a sequence of object graphs as keyframes).
- **Numpy-only feasible:** yes. `on(a,b)` checks xy-overlap, the z-gap between a's bottom and b's top, and contact. `in(a,b)` checks the center inside b's inner box. `grasped(a)` checks EE distance and gripper closure. The graph is a dict.
- **Describes the skill to a caller as:** precondition predicates, effect predicates (edge adds and deletes), role bindings, and the thresholds used by each predicate.
- **Helps when:** doing long-horizon chaining or task planning, detecting success, interfacing with language, or generalizing across layouts.
- **Fails when:** continuous quality matters (how far inside, alignment), thresholds are brittle near boundaries, or a needed relation is missing from the vocabulary.
- **How to test it:** Compare predicate-based success detection and demo segmentation with pose distance to the demo's final state. Measure agreement with the simulator's ground-truth success in fresh scenes, and the chaining success rate of 2–4 skills.
- **References:**
  - A. Rosinol, A. Gupta, M. Abate, J. Shi, L. Carlone. "3D Dynamic Scene Graphs: Actionable Spatial Perception with Places, Objects, and Humans." RSS 2020. arXiv:2002.06289
  - W. Yuan, C. Paxton, K. Desingh, D. Fox. "SORNet: Spatial Object-Centric Representations for Sequential Manipulation." CoRL 2021. arXiv:2109.03891
  - Q. Gu et al. "ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning." ICRA 2024. arXiv:2309.16650
  - Y. Zhu et al. "Vision-based Manipulation from Single Human Video with Open-World Object Graphs" (ORION). arXiv:2405.20321, 2024 (journal version: Autonomous Robots, 2026).

### A.11 Optical flow and point tracks (object motion as the interface)
- **Assumes:** A skill is captured by how points on the manipulated object should move, independent of the embodiment producing that motion. The actuator command is solved afterwards.
- **Executable form:** Represent the plan as future trajectories of K object points (2D tracks or 3D flow). Then either condition the policy on predicted tracks (ATM, Im2Flow2Act), or fit per-step rigid transforms to the tracks and map them to EE targets through the grasp transform (Track2Act, Dream2Flow).
- **Numpy-only feasible:** yes, given object poses:
  - Demo tracks are object-frame points pushed through the demo's object poses.
  - At run time, re-anchor the demo track in the current object and target frames.
  - Recover the rigid transform per step with Kabsch (`Rotation.align_vectors`).
  - `T_ee_target = T_obj_target · T_obj←ee(grasp)`, using the actual grasp rather than the demo's.
- **Describes the skill to a caller as:** object motion spec = K points × T steps in the object or world frame. It is embodiment-agnostic, requires a stable grasp or contact to realize, and handles one rigid body per part.
- **Helps when:** transferring from human or other-robot demos, when the grasp differs from the demo, or for articulated motion given per part.
- **Fails when:** the force profile matters (non-prehensile, dynamic), points are occluded or slide, or motion is dense and non-rigid.
- **How to test it:** Compare replaying the demo EE trajectory (in the object frame) with object-flow-derived EE targets. Measure success in fresh scenes where the grasp pose is randomized away from the demo grasp.
- **References:**
  - C. Wen, X. Lin, J. So, K. Chen, Q. Dou, Y. Gao, P. Abbeel. "Any-point Trajectory Modeling for Policy Learning." RSS 2024. arXiv:2401.00025
  - H. Bharadhwaj, R. Mottaghi, A. Gupta, S. Tulsiani. "Track2Act: Predicting Point Tracks from Internet Videos enables Generalizable Robot Manipulation." ECCV 2024. arXiv:2405.01527
  - M. Xu, Z. Xu, Y. Xu, C. Chi, G. Wetzstein, M. Veloso, S. Song. "Flow as the Cross-Domain Manipulation Interface" (Im2Flow2Act). CoRL 2024. arXiv:2407.15208
  - K. Dharmarajan, W. Huang, J. Wu, L. Fei-Fei, R. Zhang. "Dream2Flow: Bridging Video Generation and Open-World Manipulation with 3D Object Flow." arXiv:2512.24766, 2025.

### A.12 Spatial softmax / feature points
- **Assumes:** Task-relevant visual information reduces to a few 2D locations. A soft-argmax bottleneck turns conv feature maps into expected coordinates that a controller can use directly.
- **Executable form:** CNN → K feature maps → softmax over each map → expected `(x, y)` per map → policy MLP. It is the default pooling in robomimic and Diffusion Policy image encoders. Unsupervised keypoints come from a reconstruction objective (Kulkarni et al.).
- **Numpy-only feasible:** no (needs images/torch). The numpy analog is soft-argmax over any score grid (e.g. A.6/A.7 maps), `E[x] = Σ softmax(s/τ)·x`, which gives a smooth location estimate. With full object state, the "feature points" are simply the object positions.
- **Describes the skill to a caller as:** K learned image points with confidences. These are unnamed unless inspected, which makes them hard to expose as a caller-facing interface.
- **Helps when:** training image-based visuomotor policies from few demos, where it gives smooth coordinate-like features.
- **Fails when:** there are multiple instances (the softmax averages peaks and lands between objects), occlusions, or a caller needs named entities.
- **How to test it:** In the image setting, compare global average pooling with spatial softmax. In numpy, compare hard argmax with soft-argmax (sweep τ) over score grids in fresh scenes with 1 vs 2 valid targets, measuring success and target error.
- **References:**
  - S. Levine, C. Finn, T. Darrell, P. Abbeel. "End-to-End Training of Deep Visuomotor Policies." JMLR 17, 2016. arXiv:1504.00702
  - C. Finn, X. Y. Tan, Y. Duan, T. Darrell, S. Levine, P. Abbeel. "Deep Spatial Autoencoders for Visuomotor Learning." ICRA 2016. arXiv:1509.06113
  - T. Kulkarni, A. Gupta, C. Ionescu, et al. "Unsupervised Learning of Object Keypoints for Perception and Control." NeurIPS 2019. arXiv:1906.11883
  - A. Mandlekar, D. Xu, J. Wong, et al. "What Matters in Learning from Offline Human Demonstrations for Robot Manipulation" (robomimic). CoRL 2021. arXiv:2108.03298

### A.13 Visual cue overlays and visual prompting
- **Assumes:** Policies and VLMs use spatial relations better when those relations are rendered explicitly into the input rather than left implicit.
- **Executable form:**
  - AimBot projects the EE's approach axis ("shooting line") and a scope reticle into each camera view, using depth, extrinsics and the EE pose. The model is unchanged; only the input images are replaced.
  - TraceVLA overlays the past motion trace.
  - Set-of-Mark and PIVOT overlay numbered marks or candidate arrows, and a VLM picks one.
  - Numpy analog: append explicit relation features to the state, such as the point where the EE approach ray hits the target's box (ray–OBB intersection), the lateral miss distance of the target from the gripper axis, the distance along the axis, and a short summary of the recent EE path.
- **Numpy-only feasible:** partly. Literal overlays need images, but the numeric cue features are cheap.
- **Describes the skill to a caller as:** extra observation channels (aim point, miss distance, trace) that it expects and how they are computed (from EE pose + object geometry). Callers must supply them.
- **Helps when:** alignment is precise (grasp, insert, press), data is low, or a VLM chooses among candidates.
- **Fails when:** a miscalibrated pose or extrinsics makes the cue misleading, many marks clutter the input, or the phase is force-dominated and geometry cues add nothing.
- **How to test it:** Train with and without cue features, using the same policy class and demos. Measure success and final lateral alignment error in fresh scenes with tight-tolerance targets.
- **References:**
  - Y. Dai, J. Lee, Y. Zhang, Z. Ma, J. Yang, A. Zadeh, C. Li, N. Fazeli, J. Chai. "AimBot: A Simple Auxiliary Visual Cue to Enhance Spatial Awareness of Visuomotor Policies." CoRL 2025. arXiv:2508.08113
  - J. Yang, H. Zhang, F. Li, X. Zou, C. Li, J. Gao. "Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V." arXiv:2310.11441, 2023.
  - S. Nasiriany, F. Xia, W. Yu, T. Xiao, et al. "PIVOT: Iterative Visual Prompting Elicits Actionable Knowledge for VLMs." ICML 2024. arXiv:2402.07872
  - R. Zheng, Y. Liang, S. Huang, J. Gao, H. Daumé III, A. Kolobov, F. Huang, J. Yang. "TraceVLA: Visual Trace Prompting Enhances Spatial-Temporal Awareness for Generalist Robotic Policies." ICLR 2025. arXiv:2412.10345

### A.14 Contact and wrench representations
- **Assumes:** In contact phases, the relevant state is the contact state and the interaction wrench, not pose alone. The right structure is a task frame that splits axes into force-controlled (constrained) and position-controlled (free) ones.
- **Executable form:**
  - Choose a task frame and a per-axis mode: position, force, or stiffness value. Raibert–Craig selection matrix; ACP learns stiffness over time.
  - Add F/T and contact flags to the observation (FoAR gates F/T by predicted contact; FACTR trains the policy to attend to force).
  - Use force thresholds as guards and terminations, e.g. descend until `F_z > f*`.
  - Optionally parameterize actions as a contact location plus a post-contact motion (A.2, HACMan).
- **Numpy-only feasible:** partly. It is feasible if the sim exposes contact forces or points; otherwise infer contact from distances and sudden object-velocity changes. Impedance and hybrid laws are a few lines of numpy if the controller accepts force or stiffness targets.
- **Describes the skill to a caller as:** task frame, per-axis mode, force limits, guard conditions (contact detected, F > τ, stall), and the expected contact location.
- **Helps when:** inserting, wiping, pushing, pivoting, or when pose is uncertain and contact disambiguates it. It also adds safety.
- **Fails when:** F/T is noisy or low-rate, a wrong mode selection destabilizes control, or the sim's contact model is poor.
- **How to test it:** Compare position-only with position + force guards or compliance. Inject pose error (e.g. ±5 mm, ±5°) in fresh scenes and measure success, peak contact force and time to completion.
- **References:**
  - M. H. Raibert, J. J. Craig. "Hybrid Position/Force Control of Manipulators." ASME J. Dyn. Sys., Meas., Control 103(2):126–133, 1981. doi:10.1115/1.3139652
  - Y. Hou, Z. Liu, C. Chi, et al. "Adaptive Compliance Policy: Learning Approximate Compliance for Diffusion Guided Control." ICRA 2025. arXiv:2410.09309
  - Z. He, H. Fang, J. Chen, H.-S. Fang, C. Lu. "FoAR: Force-Aware Reactive Policy for Contact-Rich Robotic Manipulation." IEEE RA-L 2025. arXiv:2411.15753
  - J. J. Liu, Y. Li, K. Shaw, T. Tao, R. Salakhutdinov, D. Pathak. "FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning." RSS 2025. arXiv:2502.17432

### A.15 Tactile representations
- **Assumes:** Fingertip contact geometry and slip carry information vision lacks: occluded contact, grasp stability and in-hand pose.
- **Executable form:** Encode tactile images or taxel arrays with a CNN, or lift them to 3D points in the same frame as vision (3D-ViTac). Fuse them with vision for grasp-outcome prediction, slip reaction, or a fast inner reactive loop under a slow planner (RDP).
- **Numpy-only feasible:** no (needs tactile sensors, usually images/torch). A proxy is gripper width, finger-contact booleans, and drift of the object-in-hand relative pose as a slip signal that triggers a re-grasp.
- **Describes the skill to a caller as:** it requires tactile input, the signals it uses (contact patch, slip), and its reaction rate.
- **Helps when:** handling fragile objects, doing in-hand manipulation, verifying grasps, or dealing with occluded contacts.
- **Fails when:** the task is free-space motion, the sensor is specific or worn, or the sim-to-real gap is large.
- **How to test it:** Compare vision-only with vision + tactile, measuring grasp-outcome prediction accuracy and success on fragile or in-hand tasks. In numpy, compare with vs without the slip-proxy re-grasp trigger under perturbation.
- **References:**
  - R. Calandra, A. Owens, M. Upadhyaya, W. Yuan, J. Lin, E. H. Adelson, S. Levine. "The Feeling of Success: Does Touch Sensing Help Predict Grasp Outcomes?" CoRL 2017. arXiv:1710.05512
  - M. A. Lee, Y. Zhu, K. Srinivasan, et al. "Making Sense of Vision and Touch: Self-Supervised Learning of Multimodal Representations for Contact-Rich Tasks." ICRA 2019. arXiv:1810.10191
  - B. Huang, Y. Wang, X. Yang, Y. Luo, Y. Li. "3D-ViTac: Learning Fine-Grained Manipulation with Visuo-Tactile Sensing." CoRL 2024. arXiv:2410.24091
  - H. Xue, J. Ren, W. Chen, et al. "Reactive Diffusion Policy: Slow-Fast Visual-Tactile Policy Learning for Contact-Rich Manipulation." RSS 2025. arXiv:2503.02881

### A.16 Language and semantic features
- **Assumes:** Task-relevant variables can be named (object classes, parts, relations, attributes). Language, or language-aligned features, can therefore choose which state features matter and name the skill's parameters.
- **Executable form:**
  - LGA: the task description is turned into a state-abstraction mask that keeps only the relevant objects and attributes, and the policy is trained on the abstract state.
  - Code-as-Policies: the policy is code over named perception APIs and parameterized primitives.
  - With images: language-aligned embeddings (R3M) or CLIP feature fields queried by text (F3RM).
- **Numpy-only feasible:** partly. The coding agent itself plays the LM: it writes the feature mask by name, writes policy code over named objects, and names the skill's typed arguments. Embeddings need torch.
- **Describes the skill to a caller as:** a natural-language name + typed arguments (object roles, part names), the list of relevant features, and semantic preconditions.
- **Helps when:** specifying tasks, avoiding spurious correlations, giving callers human-readable interfaces, or generalizing across instructions.
- **Fails when:** language is ambiguous, the precise geometry cannot be put into words, or the LM drops a feature that was needed.
- **How to test it:** Compare all features with the language-selected feature mask. Evaluate in fresh scenes where a spurious correlate from the demos (distractor position or color) is changed, measuring success and the drop relative to in-distribution.
- **References:**
  - A. Peng, I. Sucholutsky, B. Z. Li, T. R. Sumers, T. L. Griffiths, J. Andreas, J. A. Shah. "Learning with Language-Guided State Abstractions." ICLR 2024. arXiv:2402.18759
  - J. Liang, W. Huang, F. Xia, P. Xu, K. Hausman, B. Ichter, P. Florence, A. Zeng. "Code as Policies: Language Model Programs for Embodied Control." ICRA 2023. arXiv:2209.07753
  - S. Nair, A. Rajeswaran, V. Kumar, C. Finn, A. Gupta. "R3M: A Universal Visual Representation for Robot Manipulation." CoRL 2022. arXiv:2203.12601
  - W. Shen, G. Yang, A. Yu, J. Wong, L. P. Kaelbling, P. Isola. "Distilled Feature Fields Enable Few-Shot Language-Guided Manipulation" (F3RM). CoRL 2023. arXiv:2308.07931

### A.17 Dense descriptors and correspondence fields
- **Assumes:** Points on different instances and poses of a category can be put in correspondence. A demo transfers by mapping its contact points or relative poses through that correspondence.
- **Executable form:** Learn per-pixel descriptors (Dense Object Nets) or SE(3)-equivariant descriptor fields (NDF). At test time, optimize for the gripper or object pose whose descriptors match the demo's, then execute the demo in that frame. DINOBot retrieves the most similar demo object and aligns to it with DINO features before replaying.
- **Numpy-only feasible:** partly. With known geometry, use normalized object coordinates (the object-frame point divided by the half-extents) as the descriptor. The correspondence across sizes is "same normalized coordinate", and pose alignment uses Kabsch/ICP (scipy). Learned descriptors need images/torch.
- **Describes the skill to a caller as:** a demo reference (object class, contact points in canonical coordinates) and transfer = correspondence + alignment. It is valid for same-category instances and states its symmetry assumptions.
- **Helps when:** doing one- or few-shot transfer to new instances in arbitrary poses, or handling deformables (DON).
- **Fails when:** symmetry makes correspondences ambiguous, instances differ in topology, or image descriptors face textureless surfaces.
- **How to test it:** Compare absolute object-frame offsets with normalized-coordinate correspondence on instances of varied scale and aspect ratio. Measure success in fresh scenes with new instance sizes.
- **References:**
  - P. R. Florence, L. Manuelli, R. Tedrake. "Dense Object Nets: Learning Dense Visual Object Descriptors By and For Robotic Manipulation." CoRL 2018. arXiv:1806.08756
  - P. Florence, L. Manuelli, R. Tedrake. "Self-Supervised Correspondence in Visuomotor Policy Learning." IEEE RA-L 5(2), 2020. arXiv:1909.06933
  - A. Simeonov, Y. Du, A. Tagliasacchi, J. B. Tenenbaum, A. Rodriguez, P. Agrawal, V. Sitzmann. "Neural Descriptor Fields: SE(3)-Equivariant Object Representations for Manipulation." ICRA 2022. arXiv:2112.05124
  - N. Di Palo, E. Johns. "DINOBot: Robot Manipulation via Retrieval and Alignment with Vision Foundation Models." ICRA 2024. arXiv:2402.13181

### A.18 Articulation and kinematic-structure representations
- **Assumes:** Articulated objects move along low-dimensional joints (revolute, prismatic, screw). Representing the joint axis and joint value turns opening and closing into a 1-D problem.
- **Executable form:** Estimate the joint type and axis from tracked part poses: screw decomposition of relative transforms, or a line fit for prismatic joints. The state is the joint value q, and the action is motion along the joint (the EE follows an arc about the axis while the grasp is held). Alternatively, per-point articulation flow gives the instantaneous pull direction (FlowBot3D). Screw axes can also parameterize bimanual relative motion (ScrewMimic).
- **Numpy-only feasible:** yes. Get axis and angle from `Rotation` of relative part poses, and a least-squares axis point. Classify the joint by residuals of the revolute vs prismatic fit, then generate the arc trajectory.
- **Describes the skill to a caller as:** joint type, axis (point, direction) in the object frame, current and target q, grasp on the part, and model-fit residual. It is valid while the grasp holds and the residual is below ε.
- **Helps when:** handling drawers, doors, lids and knobs, or instances with different sizes or hinge sides.
- **Fails when:** joints are multi-DoF or coupled, joint limits are unknown, the axis cannot be estimated before any motion (needs an exploratory nudge), or the grasp slips.
- **How to test it:** Compare a straight-line pull (demo replay in the object frame) with an axis-constrained arc. Measure the joint value reached and success in fresh scenes with varied door widths, hinge sides and drawer depths.
- **References:**
  - J. Sturm, C. Stachniss, W. Burgard. "A Probabilistic Framework for Learning Kinematic Models of Articulated Objects." JAIR 41:477–526, 2011. doi:10.1613/jair.3229
  - B. Eisner, H. Zhang, D. Held. "FlowBot3D: Learning 3D Articulation Flow to Manipulate Articulated Objects." RSS 2022. arXiv:2205.04382
  - Z. Jiang, C.-C. Hsu, Y. Zhu. "Ditto: Building Digital Twins of Articulated Objects from Interaction." CVPR 2022. arXiv:2202.08227
  - A. Bahety, P. Mandikal, B. Abbatematteo, R. Martín-Martín. "ScrewMimic: Bimanual Imitation from Human Videos with Screw Space Projection." RSS 2024. arXiv:2405.03666

---

# B. Learning-objective and data priors

What the policy is trained to predict or optimize, and what data it is trained on.

Rules shared by every "How to test it" below: train both arms on **all** demos. Evaluate success rate (using the task's success predicate) over the same set of ≥50 fresh simulator resets with fixed seeds for each arm. Where it matters, add resets whose object poses or parameters lie outside the demo range. Report mean ± binomial standard error.

---

### B.1 Unimodal regression behavior cloning (MSE / L1)
- **Assumes:** In each state the demonstrator's action is one value plus zero-mean noise. The conditional mean (MSE) or median (L1/Huber) is therefore a valid action, and the demos follow a single consistent strategy.
- **Executable form:** Fit `a = f(s)` by minimizing `||f(s) − a||²` (or L1/Huber) over all demo (state, action) pairs. Normalize each state and action dimension. Prefer object-frame, delta-pose or target-pose actions over raw joint torques. `f` can be ridge regression on hand-built relative features (EE − object, object − goal), kNN or RBF regression, or a small MLP.
- **Numpy-only feasible:** yes. Options: closed-form ridge (`np.linalg.solve`), kNN (`scipy.spatial.cKDTree`), or a 1–2 hidden-layer MLP trained with hand-written backprop and Adam.
- **Describes the skill to a caller as:** "Deterministic imitation of N demos. Reproduces the average demonstrated action near demonstrated states. No guarantee away from the demo state distribution."
- **Helps when / fails when:** Helps with few, consistent demos (a single scripted or planner expert) in low-dimensional state. Fails when demos mix strategies: it averages two valid modes into an invalid action (e.g., straight into an obstacle both modes avoided). Also fails when errors compound over long horizons.
- **How to test it:** Compare MSE, L1 and a multimodal head (B.2) on the same features. Diagnostic: for demo states whose nearest neighbours in state space are close, measure the spread of their actions. A high spread predicts that MSE will fail.
- **References:**
  - Pomerleau, "ALVINN: An Autonomous Land Vehicle in a Neural Network," NeurIPS (NIPS) 1988. https://proceedings.neurips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html
  - Mandlekar et al., "What Matters in Learning from Offline Human Demonstrations for Robot Manipulation," CoRL 2021. arXiv:2108.03298
  - Foster et al., "Is Behavior Cloning All You Need? Understanding Horizon in Imitation Learning," NeurIPS 2024. arXiv:2407.15007

### B.2 Multimodal explicit behavior cloning (mixture density, discretized / clustered actions)
- **Assumes:** `p(a|s)` has several distinct modes (grasp from the left or the right, different subtask orders, detour direction). The policy should commit to one mode rather than average them.
- **Executable form:** Three variants:
  - A K-component Gaussian mixture head trained by negative log-likelihood (MDN).
  - Per-dimension action bins with a classification loss (RT-1 uses 256 bins).
  - k-means over demo actions, then predict the cluster (cross-entropy) plus a per-cluster continuous offset (BeT). VQ-BeT uses learned residual-VQ codes instead of k-means.

  At run time, sample a mode or take the most likely one, and keep it fixed for a chunk or segment so the policy does not flip between modes.
- **Numpy-only feasible:** yes. BeT-style: `scipy.cluster.vq.kmeans2` on actions, a softmax/multinomial logistic classifier, and per-cluster ridge offsets. Alternatively, Gaussian mixture regression: fit a GMM to joint `(s, a)` by EM in numpy, then condition on `s` in closed form.
- **Describes the skill to a caller as:** "Action distribution has K modes found in the demos (e.g., approach from the left or from the right). Commits to one mode per episode or chunk. The caller can pin a mode if one is exposed."
- **Helps when / fails when:** Helps with human or multi-strategy demos. Fails when each mode has too few demos (clusters overfit). Also fails when a mode is resampled independently every step, which causes jitter between modes.
- **How to test it:** MSE head vs clustered head vs GMM head, with identical inputs. Metrics: success rate, fraction of executed actions lying between clusters (mode-averaging), and mode switches per episode.
- **References:**
  - Bishop, "Mixture Density Networks," Technical Report NCRG/94/004, Aston University, 1994. https://publications.aston.ac.uk/id/eprint/373/
  - Shafiullah et al., "Behavior Transformers: Cloning k modes with one stone," NeurIPS 2022. arXiv:2206.11251
  - Brohan et al., "RT-1: Robotics Transformer for Real-World Control at Scale," RSS 2023. arXiv:2212.06817
  - Lee et al., "Behavior Generation with Latent Actions" (VQ-BeT), ICML 2024. arXiv:2403.03181

### B.3 Diffusion and flow-matching action objectives
- **Assumes:** The distribution of actions or action chunks is complex and multimodal but smooth. Learning to denoise noisy actions, conditioned on state, captures it more stably than mixtures or energy models.
- **Executable form:** Two objectives:
  - DDPM: train `ε_θ(a_k, k, s)` to predict the noise added at level k.
  - Flow matching / rectified flow: sample `ε ~ N(0,I)` and `t ~ U[0,1]`, form `a_t = t·a + (1−t)·ε`, and regress `v_θ(a_t, t, s)` onto `a − ε`.

  At run time, integrate from Gaussian noise in 5–100 steps (DDIM or Euler). The target is usually an action chunk (B.5).
- **Numpy-only feasible:** partly. Flow matching is plain MSE regression on `(s, a_t, t) → a − ε`, so a small numpy MLP can learn it for low-dimensional actions or short chunks, but training is slow and needs careful tuning. High-dimensional chunks or large datasets need torch.
- **Describes the skill to a caller as:** "Generative imitation of the demo action distribution. Output is stochastic and can represent several modes. Each action costs K network evaluations."
- **Helps when / fails when:** Helps with multimodal, high-dimensional or chunked actions and roughly ≥50 demos. Fails with very few demos (memorizes them) and under tight latency budgets. Where unimodal regression already solves the task, it only adds cost.
- **How to test it:** Flow-matching head vs MSE head vs clustered head (B.2) with the same inputs and data. Metrics: success rate, wall-clock time per action, and success vs number of demos (5/20/100).
- **References:**
  - Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion," RSS 2023 (extended version in IJRR). arXiv:2303.04137
  - Lipman et al., "Flow Matching for Generative Modeling," ICLR 2023. arXiv:2210.02747
  - Reuss et al., "Goal-Conditioned Imitation Learning using Score-based Diffusion Policies" (BESO), RSS 2023. arXiv:2304.02532
  - Black et al., "π0: A Vision-Language-Action Flow Model for General Robot Control," RSS 2025. arXiv:2410.24164

### B.4 Energy-based / implicit behavior cloning
- **Assumes:** A scalar compatibility `E(s, a)` whose minimum is the expert action is easier to learn than a direct map `s → a`. This handles discontinuous switching (make or break contact, left vs right) and multimodality.
- **Executable form:** Train `E_θ(s, a)` with InfoNCE: the expert action against N negative actions sampled uniformly within the action bounds. Act with `a = argmin_a E(s, a)` using derivative-free optimization: sample, keep the elites, resample with shrinking noise (CEM). Langevin sampling is the alternative.
- **Numpy-only feasible:** yes, for low-dimensional actions (≤ ~6-D). Train a small numpy MLP energy with softmax cross-entropy over (1 positive + N negatives), and run CEM over 256–1024 samples at inference. Simplest version: a nonparametric energy equal to the distance to the nearest demo `(s, a)` pairs in a KD-tree.
- **Describes the skill to a caller as:** "Picks the lowest-energy action within bounds [lo, hi]. Can represent discontinuous switches. Each step runs an N-sample × I-iteration optimization."
- **Helps when / fails when:** Helps with discontinuous or multimodal low-dimensional actions. Fails in high action dimensions (uniform negatives stop being informative), when training is unstable, and when inference compute is limited.
- **How to test it:** Implicit (CEM argmin) vs explicit MSE, with equal model capacity, on a task with a discontinuity. Metrics: success rate. Also sweep the number of negatives and CEM iterations.
- **References:**
  - Florence et al., "Implicit Behavioral Cloning," CoRL 2021. arXiv:2109.00137
  - Ta et al., "Conditional Energy-Based Models for Implicit Policies: The Gap between Theory and Practice," arXiv 2022 (RSS 2022 workshop). arXiv:2207.05824
  - Singh et al., "Revisiting Energy Based Models as Policies: Ranking Noise Contrastive Estimation and Interpolating Energy Models," arXiv 2023. arXiv:2309.05803

### B.5 Action chunking and temporal ensembling
- **Assumes:** Expert actions are temporally correlated and partly non-Markovian (pauses, hesitations). Predicting H future actions per query shortens the effective decision horizon, reduces compounding error, and gives smooth, committed motion.
- **Executable form:** Train `f(s_t) → [a_t, …, a_{t+H−1}]` as a flattened regression or generative target. Three ways to execute:
  - Run the first k ≤ H actions, then re-query (receding horizon).
  - Query every step and average all predictions made for time t with weights `w_i = exp(−m·i)` (ACT temporal ensembling).
  - Sample several chunks and keep the one most consistent with the previous chunk (BID).

  For RL, learn the critic over chunks (Q-chunking).
- **Numpy-only feasible:** yes. A chunk only widens the regression output to `H·d_a`, and ensembling is a weighted average.
- **Describes the skill to a caller as:** "Commits to H-step action plans and replans every k steps, so its reaction latency is k·dt. Smooth motion, but it will not react to disturbances inside a chunk."
- **Helps when / fails when:** Helps with noisy or pausing demos, fine precise motions, and committing to one mode. Fails when object state changes faster than k steps (open-loop drift). Ensembling averages across modes when successive queries pick different modes.
- **How to test it:** Sweep H ∈ {1, 4, 8, 16} and k ∈ {1, H/2, H}, each with and without ensembling. Metrics: success rate, action jerk, and success when the object is displaced mid-episode.
- **References:**
  - Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (ACT), RSS 2023. arXiv:2304.13705
  - Liu et al., "Bidirectional Decoding: Improving Action Chunking via Guided Test-Time Sampling," ICLR 2025. arXiv:2408.17355
  - Li et al., "Reinforcement Learning with Action Chunking" (Q-chunking), NeurIPS 2025. arXiv:2507.07969
  - Lazzati et al., "Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?," arXiv 2026. arXiv:2608.02547

### B.6 Contrastive, time-contrastive and progress/value objectives
- **Assumes:** A useful representation places states closer together as they get closer in time or closer to the goal. The temporal order of a demo is free supervision for task progress.
- **Executable form:** Three variants:
  - Time-contrastive: train an embedding `φ(s)` in which temporally near states attract and far ones repel (triplet or InfoNCE loss).
  - Goal-conditioned contrastive: `f(s, a, g) = φ(s, a)ᵀψ(g)` trained with InfoNCE, using future states of the same trajectory as positives. This critic is a goal-conditioned Q-function.
  - Progress regression: predict the normalized time t/T of each demo state (or a VIP-style value).

  Use `−d(φ(s), φ(g))` or predicted progress as a dense reward, a success or termination signal, or a stall detector.
- **Numpy-only feasible:** yes. Progress regression by ridge or kNN on object-centric features. A linear or small-MLP InfoNCE embedding in numpy. No images needed.
- **Describes the skill to a caller as:** "Exposes a progress estimate in [0, 1] learned from demos. The caller can monitor it to detect stalls or failures and trigger a retry."
- **Helps when / fails when:** Helps when demos progress monotonically toward the goal, and for creating dense rewards for RL fine-tuning. Fails on non-monotone tasks (retreat, regrasp, cyclic motion), on demos with variable speed (time ≠ progress), and off-distribution, where estimates are overconfident.
- **How to test it:** Spearman correlation between predicted progress and true task phase on fresh rollouts. AUROC for detecting failed rollouts. RL with progress reward vs sparse reward, comparing success vs environment steps.
- **References:**
  - Sermanet et al., "Time-Contrastive Networks: Self-Supervised Learning from Video," ICRA 2018. arXiv:1704.06888
  - Ma et al., "VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training," ICLR 2023. arXiv:2210.00030
  - Eysenbach et al., "Contrastive Learning as Goal-Conditioned Reinforcement Learning," NeurIPS 2022. arXiv:2206.07568
  - Ma et al., "Vision Language Models are In-Context Value Learners" (GVL), ICLR 2025. arXiv:2411.04549

### B.7 Dynamics-prediction auxiliaries (forward / future-state / video, inverse dynamics)
- **Assumes:** Features that predict how the world changes under an action (forward), or which action explains a transition (inverse), are the features control needs. State-only data can be given action labels by an inverse model.
- **Executable form:** Add loss terms on a trunk shared with the policy:
  - Forward: `||f(s_t, a_t) − s_{t+1}||²`, or predict `s_{t+k}` or future frames (GR-1).
  - Inverse: `||g(s_t, s_{t+1}) − a_t||²`.

  Alternatively, use `g` to label state-only demos (BCO), or predict the next object state and then act through the inverse model (predictive inverse dynamics, Seer). With a simulator, fit `f` on cheap reset/step transitions and use it for short-horizon MPC or to sanity-check policy outputs.
- **Numpy-only feasible:** yes. Ridge, local-linear or small-MLP forward and inverse models on object-centric state, with transitions from sim rollouts.
- **Describes the skill to a caller as:** "Policy = predicted next object state + inverse model. The caller receives the expected outcome (e.g., next object pose) and can compare it with the observation to detect failure."
- **Helps when / fails when:** Helps when demos are few but transitions are plentiful, and when demos lack action labels. Fails when the inverse is ambiguous (without contact the object does not move for any action), and in chaotic contact dynamics where one-step errors compound.
- **How to test it:**
  - Policy with vs without the auxiliary loss: success rate.
  - Inverse-model labeling: action-label error against true actions, and BC success using inferred labels vs true labels.
  - Forward model: k-step open-loop prediction error.
- **References:**
  - Agrawal et al., "Learning to Poke by Poking: Experiential Learning of Intuitive Physics," NeurIPS 2016. arXiv:1606.07419
  - Torabi et al., "Behavioral Cloning from Observation," IJCAI 2018. arXiv:1805.01954
  - Wu et al., "Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation" (GR-1), ICLR 2024. arXiv:2312.13139
  - Tian et al., "Predictive Inverse Dynamics Models are Scalable Learners for Robotic Manipulation" (Seer), ICLR 2025. arXiv:2412.15109

### B.8 Target, keyframe and subgoal prediction (grasp pose, contact point, goal pose, waypoints)
- **Assumes:** Manipulation is decided by a few sparse targets: where to grasp, where to place, where to make contact, and the next bottleneck pose. Predicting these and letting a controller or planner connect them generalizes better than dense per-step actions.
- **Executable form:**
  1. Extract targets from demos: EE pose at gripper close/open events, object pose at release, first-contact point, and AWE waypoints (the smallest subset of trajectory points whose piecewise-linear interpolation reconstructs the trajectory within ε).
  2. Learn `target = h(object state)`, expressed in the object frame.
  3. Execute with IK, a motion planner or a P-controller between targets.

  Subgoal variant: predict the next intermediate state and pass it to a goal-reaching low-level policy.
- **Numpy-only feasible:** yes. AWE is dynamic programming or recursive search over trajectory points with an error threshold. Target regression in the object frame uses ridge or kNN. IK uses `scipy.optimize`. Execution is P-control.
- **Describes the skill to a caller as:** "Skill = predicted target sequence (pregrasp → grasp → lift → place) in the object frame, with planner-generated motion between targets. Reliable for object poses within the demo range."
- **Helps when / fails when:** Helps with few demos, long horizons and spatial generalization. Fails when the task needs continuous contact behavior between keyframes (wiping, pushing along a path, pouring), and when targets are ambiguous (symmetric objects).
- **How to test it:** Dense per-step BC vs keyframe prediction + controller. Metric: success on fresh resets, including object poses outside the demo range. Also sweep the AWE threshold ε (number of waypoints) against success.
- **References:**
  - Zeng et al., "Transporter Networks: Rearranging the Visual World for Robotic Manipulation," CoRL 2020. arXiv:2010.14406
  - Mo et al., "Where2Act: From Pixels to Actions for Articulated 3D Objects," ICCV 2021. arXiv:2101.02692
  - Shi et al., "Waypoint-Based Imitation Learning for Robotic Manipulation" (AWE), CoRL 2023. arXiv:2307.14326
  - Black et al., "Zero-Shot Robotic Manipulation with Pretrained Image-Editing Diffusion Models" (SuSIE), ICLR 2024. arXiv:2310.10639

### B.9 Reward shaping (potential-based, staged, LLM-written reward code)
- **Assumes:** The designer knows the task's intermediate structure (reach → grasp → lift → place) and can encode it as a dense reward. This speeds up RL, and a potential-based form leaves the optimal policy unchanged.
- **Executable form:** Three forms:
  - Potential-based: `r' = r + γΦ(s') − Φ(s)` with, for example, `Φ = stage_index − normalized distance to the current stage's target`.
  - Staged rewards: a bonus on completing each stage.
  - LLM-written reward code (Eureka / Text2Reward): an LLM writes the reward from the state API and task text, the agent runs RL, returns per-component reward statistics to the LLM, and the loop repeats.
- **Numpy-only feasible:** yes. The reward is a Python function of state. The optimizer can be CEM, CMA-ES or random search over the parameters of a linear/MLP policy or a scripted controller, or a small numpy actor-critic.
- **Describes the skill to a caller as:** "Optimized for shaped reward terms [...] with weights [...]. It optimizes those terms, which may differ from the success predicate. Success-predicate rate: p."
- **Helps when / fails when:** Helps on sparse-reward, long-horizon tasks with clear geometric progress. Fails through reward hacking when shaping is not potential-based (e.g., hovering near the object to collect reach reward), and through local optima when stage terms conflict.
- **How to test it:** Sparse reward vs potential-based vs non-potential shaping. Metric: true success rate vs environment steps. Log the gap between shaped return and success to detect hacking.
- **References:**
  - Ng, Harada, Russell, "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping," ICML 1999. https://dl.acm.org/doi/10.5555/645528.657613
  - Gupta et al., "Unpacking Reward Shaping: Understanding the Benefits of Reward Engineering on Sample Complexity," NeurIPS 2022. arXiv:2210.09579
  - Ma et al., "Eureka: Human-Level Reward Design via Coding Large Language Models," ICLR 2024. arXiv:2310.12931
  - Xie et al., "Text2Reward: Reward Shaping with Language Models for Reinforcement Learning," ICLR 2024. arXiv:2309.11489

### B.10 Goal-conditioned hindsight relabeling (HER, GCSL, play data)
- **Assumes:** Any trajectory is an optimal demonstration of reaching the states it actually reached, so a failed attempt at goal g is a success for some other goal g'.
- **Executable form:**
  1. Store full trajectories.
  2. For each sampled transition, replace the goal with a future state of the same trajectory ("future" strategy, k = 4 relabels per transition).
  3. Either recompute `r(s', g')` for off-policy RL (HER), or run goal-conditioned BC on `(s, g', a)` (GCSL / play).
  4. Repeat collect → relabel → fit.
  5. At test time, condition on the requested goal, e.g. a target object pose.
- **Numpy-only feasible:** yes. Relabeling is array indexing. GCSL uses ridge, kNN or an MLP on `[s, g − s]`.
- **Describes the skill to a caller as:** "Goal-conditioned: accepts a target object pose g. Competent for goals inside the region covered by its data (report the covered range)."
- **Helps when / fails when:** Helps with sparse-reward goal reaching, multi-goal skills, and reusing failures and play data. Fails when goals are relational or semantic rather than states ("stacked", "inside"), and when reaching g by an unintended route still counts as success (e.g., moving the wrong object).
- **How to test it:** With vs without relabeling, same environment-step budget. Metrics: success on freshly sampled goals, and success as a function of goal distance from the data.
- **References:**
  - Andrychowicz et al., "Hindsight Experience Replay," NeurIPS 2017. arXiv:1707.01495
  - Ghosh et al., "Learning to Reach Goals via Iterated Supervised Learning" (GCSL), ICLR 2021. arXiv:1912.06088
  - Lynch et al., "Learning Latent Plans from Play," CoRL 2019. arXiv:1903.01973
  - Reuss et al., "Goal-Conditioned Imitation Learning using Score-based Diffusion Policies," RSS 2023. arXiv:2304.02532

### B.11 Learned rewards from demonstrations (IRL, adversarial imitation, SQIL)
- **Assumes:** The demonstrator is near-optimal for an unknown reward. Recovering that reward, or matching the demonstrator's state-action occupancy through interaction, generalizes better than copying actions and corrects covariate shift.
- **Executable form:** Four variants:
  - GAIL: a discriminator `D(s, a)` separates expert from policy samples. The policy is trained by RL on reward `−log(1 − D)`, where D is the probability of expert.
  - AIRL: the discriminator is structured as `r(s) + γh(s') − h(s)`, giving a transferable state reward.
  - SQIL: demo transitions go in the replay buffer with r = 1, own transitions with r = 0, then run soft Q-learning.
  - Rank2Reward: learn to rank states by temporal position in demos and use the ranking as a shaped reward.
- **Numpy-only feasible:** partly. The discriminator can be logistic regression on object-centric features (`scipy.optimize`), with the policy improved by CEM or ES over controller parameters. SQIL with linear or fitted-Q in low dimensions is feasible. Deep GAIL + PPO effectively needs torch.
- **Describes the skill to a caller as:** "Trained to match the demonstrators' state distribution under interaction. Recovers from off-demo states better than plain BC. Its reward is a learned discriminator, not the task predicate."
- **Helps when / fails when:** Helps with few demos plus simulator access, and against covariate shift. Fails with unstable adversarial training, discriminators that key on spurious features (time index, object id), and large environment-step budgets.
- **How to test it:** BC vs BC → SQIL/GAIL fine-tuning on the same demos. Metrics: success over fresh resets, success from perturbed initial states, and environment steps used.
- **References:**
  - Ho & Ermon, "Generative Adversarial Imitation Learning," NeurIPS 2016. arXiv:1606.03476
  - Fu et al., "Learning Robust Rewards with Adversarial Inverse Reinforcement Learning," ICLR 2018. arXiv:1710.11248
  - Reddy et al., "SQIL: Imitation Learning via Reinforcement Learning with Sparse Rewards," ICLR 2020. arXiv:1905.11108
  - Yang et al., "Rank2Reward: Learning Shaped Reward Functions from Passive Video," ICRA 2024. arXiv:2404.14735

### B.12 Success classifiers, VLM success detectors and programmatic success predicates
- **Assumes:** Success is easier to recognize than to specify. A classifier over outcome states, a VLM judging the scene, or a code predicate can serve as reward, termination and retry trigger.
- **Executable form:** Classifier route:
  1. Positives: demo final states. Negatives: initial states, mid-trajectory states, and policy states.
  2. Fit `p(success | s)`.
  3. As in VICE, keep adding the policy's own states as negatives so the policy cannot exploit the classifier.

  VLM route: query for success or preferences (needs images). In the sandbox, the agent instead writes a predicate over object state and validates it: it must be true on every demo end state and false on every demo start state.
- **Numpy-only feasible:** yes for the classifier (logistic regression, kNN, Gaussian discriminant) and the code predicate. No for the VLM variants: no images.
- **Describes the skill to a caller as:** "Success = predicate X (e.g., `|cube_z − target_z| < 1 cm and gripper open`), validated on N demo end states and N start states. The caller can invoke it to check the outcome and retry."
- **Helps when / fails when:** Helps when no reward is given, and for termination and retry logic. Fails when false positives get exploited (the object is lifted only briefly) and when negatives miss near-miss states.
- **How to test it:** Precision and recall of the predicate or classifier on demo terminal vs non-terminal states and on manually checked rollouts. Also RL with the learned predicate vs the ground-truth predicate: success rate and number of exploited false positives.
- **References:**
  - Singh et al., "End-to-End Robotic Reinforcement Learning without Reward Engineering," RSS 2019. arXiv:1904.07854
  - Du et al., "Vision-Language Models as Success Detectors," CoLLAs 2023. arXiv:2303.07280
  - Rocamonde et al., "Vision-Language Models are Zero-Shot Reward Models for Reinforcement Learning," ICLR 2024. arXiv:2310.12921
  - Wang et al., "RL-VLM-F: Reinforcement Learning from Vision Language Foundation Model Feedback," ICML 2024. arXiv:2402.03681

### B.13 Observation and geometric data augmentation
- **Assumes:** The correct action is invariant or equivariant to known nuisance transformations: pixel shifts, color, small state noise, rigid transforms of the whole scene, or changes to task-irrelevant or non-interacting objects.
- **Executable form:**
  - Images: random shift (pad 4 px, then random crop) and color jitter (RAD/DrQ).
  - State: Gaussian noise and random amplitude scaling (RAD).
  - Geometric: apply an SE(2)/SE(3) transform T to all object and EE poses and transform the actions by the same T (equivariance).
  - Counterfactual: recombine components of different transitions that do not interact locally (CoDA), or resample poses of task-irrelevant objects and rigidly transform the relevant ones (RoCoDA).
- **Numpy-only feasible:** yes for all state-space variants (array operations and `scipy.spatial.transform`). Pixel augmentations do not apply.
- **Describes the skill to a caller as:** "Trained with yaw ±π about the table center, translation ±10 cm, and resampled distractor poses. Invariance holds only for these transforms."
- **Helps when / fails when:** Helps with small datasets and known symmetries. Fails when the claimed invariance is false: joint limits, workspace bounds and gravity break SE(3) equivariance, and "irrelevant" distractors can in fact collide.
- **How to test it:** With vs without each augmentation, on fresh resets outside the demo pose range. Validity check: replay augmented trajectories open-loop in the simulator; the fraction that succeed estimates how often the invariance actually holds.
- **References:**
  - Laskin et al., "Reinforcement Learning with Augmented Data" (RAD), NeurIPS 2020. arXiv:2004.14990
  - Kostrikov et al., "Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels" (DrQ), ICLR 2021. arXiv:2004.13649
  - Pitis et al., "Counterfactual Data Augmentation using Locally Factored Dynamics" (CoDA), NeurIPS 2020. arXiv:2007.02863
  - Ameperosa et al., "RoCoDA: Counterfactual Data Augmentation for Data-Efficient Robot Learning from Demonstrations," ICRA 2025. arXiv:2411.16959

### B.14 Noise injection and corrective trajectory augmentation
- **Assumes:** Policy errors push the robot off the demo manifold. Training on states slightly off the demos, labeled with actions that lead back, prevents compounding error.
- **Executable form:** Four variants:
  - DART: while collecting with a queryable expert, execute the expert action plus noise (σ matched to the learner's error) but record the clean expert action.
  - ChauffeurNet: perturb demo trajectories and synthesize corrective labels.
  - CCIL: fit a locally Lipschitz dynamics model and generate labels that return to demo states.
  - SPARTN: the image equivalent, using novel-view synthesis.
- **Numpy-only feasible:** yes. DART with a scripted or planner expert in sim is a few lines of code. For CCIL-style labels, either fit local linear dynamics (ridge) and solve for the corrective action, or, if the sim can reset to a given state, query the expert from the perturbed state.
- **Describes the skill to a caller as:** "Trained on demo states perturbed by σ = 1 cm with corrective labels. Recovers from displacements of about σ."
- **Helps when / fails when:** Helps when an expert can be queried or dynamics are locally smooth. Fails with discontinuous contact dynamics (corrective labels are wrong), noise large enough to change the task outcome, and fixed human demos whose labels cannot be regenerated.
- **How to test it:** Clean BC vs noise-injected or perturbed data. Metrics: success on fresh resets, and success after a mid-episode object or EE displacement. Sweep σ.
- **References:**
  - Laskey et al., "DART: Noise Injection for Robust Imitation Learning," CoRL 2017. arXiv:1703.09327
  - Bansal et al., "ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst," RSS 2019. arXiv:1812.03079
  - Zhou et al., "NeRF in the Palm of Your Hand: Corrective Augmentation for Robotics via Novel-View Synthesis" (SPARTN), CVPR 2023. arXiv:2301.08556
  - Ke et al., "CCIL: Continuity-based Data Augmentation for Corrective Imitation Learning," ICLR 2024. arXiv:2310.12972

### B.15 Domain randomization
- **Assumes:** The test condition is one draw from a distribution over physical and scene parameters. A policy trained across a wide enough distribution works for any member.
- **Executable form:** At each reset, sample parameters from ranges: object size, mass, friction, damping, initial poses, control latency, observation noise. Train on the mixture. Two ways to set the ranges:
  - ADR: widen each range while success stays above a threshold.
  - DrEureka: an LLM proposes the ranges.
- **Numpy-only feasible:** yes, provided the simulator exposes these parameters at reset. Otherwise only initial-state randomization is possible.
- **Describes the skill to a caller as:** "Trained on friction 0.5–1.2, mass 0.1–0.5 kg, object size 2–6 cm. Reliable inside that box, untested outside."
- **Helps when / fails when:** Helps with robustness and sim-to-sim or sim-to-real transfer. Fails when ranges are too wide (conservative behavior, lower nominal success), when they miss the true value, and when a parameter is unobservable but decisive (then add history or system identification).
- **How to test it:** Nominal vs randomized training. Evaluate on a grid of parameter values, including out-of-range values, and report success per bin. These bins become the reliable range stated to the caller.
- **References:**
  - Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World," IROS 2017. arXiv:1703.06907
  - Peng et al., "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization," ICRA 2018. arXiv:1710.06537
  - OpenAI et al., "Solving Rubik's Cube with a Robot Hand" (ADR), arXiv 2019. arXiv:1910.07113
  - Ma et al., "DrEureka: Language Model Guided Sim-To-Real Transfer," RSS 2024. arXiv:2406.01967

### B.16 Demo-based data generation (MimicGen family)
- **Assumes:** A demo splits into object-centric subtask segments. Within each segment the EE trajectory is rigidly attached to the frame of the object being manipulated, so it can be moved to new object poses, stitched together, and validated in sim.
- **Executable form:**
  1. Split each source demo at subtask boundaries (grasp and release events).
  2. Express EE poses in the frame of the reference object.
  3. For a new scene: pick a source segment, transform it to the new object pose, and interpolate from the current EE pose.
  4. Execute in sim and keep only successful episodes.

  SkillMimicGen replaces the free-space segments with motion-planner paths. DemoGen adapts actions TAMP-style from a single demo and edits point clouds.
- **Numpy-only feasible:** yes. SE(3) algebra with numpy and `scipy.spatial.transform`, replay through sim step, and a success predicate as the filter.
- **Describes the skill to a caller as:** "Trained on N generated demos from M source demos over object-pose range R. Generation success was p per subtask; a low p marks segments that are not rigidly object-centric."
- **Helps when / fails when:** Helps with few source demos and large pose variation. Fails when a segment depends on several objects' relative poses or on dynamics (deformables, throwing), when interpolated transit segments collide, and when object geometry changes.
- **How to test it:** BC on source demos only vs on generated data at 10, 100 and 1000 episodes. Metrics: success on fresh resets over the widened pose range, and per-subtask generation success.
- **References:**
  - Mandlekar et al., "MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations," CoRL 2023. arXiv:2310.17596
  - Garrett et al., "SkillMimicGen: Automated Demonstration Generation for Efficient Skill Learning and Deployment," CoRL 2024. arXiv:2410.18907
  - Jiang et al., "DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning," ICRA 2025. arXiv:2410.24185
  - Xue et al., "DemoGen: Synthetic Demonstration Generation for Data-Efficient Visuomotor Policy Learning," RSS 2025. arXiv:2502.16932

### B.17 Synthetic expert data from planners, trajectory optimization and privileged / scripted experts
- **Assumes:** In simulation, an expert with privileged information (full state, model, simulator resets) or unlimited compute can solve each instance. The policy only has to amortize or distill that expert onto its run-time inputs.
- **Executable form:**
  1. For each sampled initial state, produce an expert solution with a scripted controller, a motion or TAMP planner, or trajectory optimization (CEM/MPPI/iLQR through sim reset/step).
  2. Keep the successes.
  3. Distill with BC, preferably plus DAgger with the same expert (B.18).

  Variants: GPS alternates trajectory optimization with policy fitting under mutual regularization. Learning by Cheating trains a privileged agent first, then supervises the student on-policy. In "Scaling Up and Distilling Down", an LLM plans and sampling-based planners generate the data, which is then distilled.
- **Numpy-only feasible:** yes. CEM/MPPI over action sequences with sim resets (needs state save/restore or deterministic replay), `scipy.optimize` for IK or trajectory optimization, and scripted experts written as code.
- **Describes the skill to a caller as:** "Distilled from expert X, which succeeds at rate p_expert on the same distribution. The expert's success is the ceiling, and the skill inherits its style (e.g., straight-line, fast motions)."
- **Helps when / fails when:** Helps when a simulator is available and demos are scarce. Fails when the expert relies on information the policy lacks (imitation gap), and when the expert's solutions are inconsistent between nearby states (random planner seeds), which a unimodal loss cannot fit.
- **How to test it:** BC from the original demos vs expert-generated data vs both, at equal data size. Metrics: success on fresh resets and the student-expert gap. Diagnostic: action variance between nearby states.
- **References:**
  - Levine & Koltun, "Guided Policy Search," ICML 2013. https://proceedings.mlr.press/v28/levine13.html
  - Chen et al., "Learning by Cheating," CoRL 2019. arXiv:1912.12294
  - Dalal et al., "Imitating Task and Motion Planning with Visuomotor Transformers" (OPTIMUS), CoRL 2023. arXiv:2305.16309
  - Ha et al., "Scaling Up and Distilling Down: Language-Guided Robot Skill Acquisition," CoRL 2023. arXiv:2307.14535

### B.18 Interactive correction (DAgger, HG-DAgger)
- **Assumes:** An expert can label the states the learner itself visits. Training on the learner's own state distribution removes compounding error (no-regret reduction).
- **Executable form:** Iterate:
  1. Roll out the current policy, optionally mixed with the expert with probability β.
  2. Get expert labels, either at every visited state (DAgger) or only where the expert takes over, recording the takeover segments (HG-DAgger, Sirius).
  3. Aggregate the new data with the old and refit.

  In sim, the expert can be a scripted controller, a planner, or a MimicGen-style replayer.
- **Numpy-only feasible:** yes. The labeler is code or a planner, and aggregating and refitting are trivial.
- **Describes the skill to a caller as:** "Trained for K rounds on its own visited states, labeled by expert X. Robust to its own drift wherever the expert is competent."
- **Helps when / fails when:** Helps whenever an expert can be queried at arbitrary states. Fails when only fixed human demos exist, when the expert breaks down from the learner's odd states (a planner failing from a near-collision), and when expert labels are inconsistent across rounds.
- **How to test it:** BC vs DAgger at equal numbers of expert labels. Metric: success on fresh resets as a function of label count.
- **References:**
  - Ross et al., "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" (DAgger), AISTATS 2011. arXiv:1011.0686
  - Kelly et al., "HG-DAgger: Interactive Imitation Learning with Human Experts," ICRA 2019. arXiv:1810.02890
  - Liu et al., "Robot Learning on the Job: Human-in-the-Loop Autonomy and Learning During Deployment" (Sirius), RSS 2023. arXiv:2211.08416

### B.19 Self-supervised data collection and success-filtered self-improvement
- **Assumes:** The robot can produce its own training data by trying things, given an automatic outcome label. Its successful attempts can then be used as demonstrations (filtered BC / self-imitation).
- **Executable form:** Two loops:
  - Self-improvement: run the current policy with exploration noise over many sim resets, label each rollout with the success predicate (B.12), add successes (optionally goal-relabeled, B.10) to the dataset, and refit.
  - Candidate scoring: sample candidate grasps or placements, execute them, and train a success predictor `p(success | s, candidate)`, then pick the argmax at run time.

  QT-Opt scales up the RL form. RoboCat retrains a generalist on data generated by its own fine-tuned agents.
- **Numpy-only feasible:** yes. Rollouts, filtering and refitting. The success predictor is logistic regression over `(s, candidate)` features.
- **Describes the skill to a caller as:** "Self-improved over R rounds of its own filtered sim attempts. Success rose from p0 to pR on the training distribution (numbers stated per initial-state bin)."
- **Helps when / fails when:** Helps when the success predicate is reliable and the initial policy succeeds sometimes. Fails when initial success is zero (nothing to keep), when predicate false positives reinforce wrong behavior, and when diversity collapses onto easy initial states.
- **How to test it:** One round vs R rounds on the same fixed evaluation seeds. Report success per round, stratified by initial-state difficulty to catch collapse.
- **References:**
  - Pinto & Gupta, "Supersizing Self-supervision: Learning to Grasp from 50K Tries and 700 Robot Hours," ICRA 2016. arXiv:1509.06825
  - Kalashnikov et al., "QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation," CoRL 2018. arXiv:1806.10293
  - Oh et al., "Self-Imitation Learning," ICML 2018. arXiv:1806.05635
  - Bousmalis et al., "RoboCat: A Self-Improving Generalist Agent for Robotic Manipulation," TMLR 2023. arXiv:2306.11706

### B.20 Curriculum (reverse / start-from-demo-states, jump-start, automatic)
- **Assumes:** The task becomes learnable if training starts from easy instances (near the goal, small perturbations, simple parameters) and difficulty grows with competence.
- **Executable form:** Four variants:
  - Demo-state starts (Salimans & Chen): reset to the demo state at time T−k and increase k once success exceeds a threshold.
  - Reverse curriculum (Florensa): generate start states by random-walking back from the goal, and keep those with intermediate success (0.1–0.9).
  - Jump-start (JSRL): a guide policy (demo replay or BC) runs the first h steps and the learner runs the rest; decrease h over training.
  - Parameter curriculum: widen parameter ranges automatically (ADR, B.15).
- **Numpy-only feasible:** yes. Demo-state and reverse curricula need a simulator that can be reset to a given state. JSRL needs only reset/step.
- **Describes the skill to a caller as:** "Learned through a start-state curriculum. Final competence is measured from the nominal initial-state distribution (success p). Earlier curriculum stages are not a claim about it."
- **Helps when / fails when:** Helps sparse-reward, long-horizon RL. Fails when the sim cannot set arbitrary states, when strategies learned from easy starts do not carry over to far starts, and when the threshold schedule stalls.
- **How to test it:** RL from nominal starts vs with the curriculum, at the same step budget. Metric: success from nominal starts vs environment steps.
- **References:**
  - Florensa et al., "Reverse Curriculum Generation for Reinforcement Learning," CoRL 2017. arXiv:1707.05300
  - Salimans & Chen, "Learning Montezuma's Revenge from a Single Demonstration," NeurIPS 2018 Deep RL Workshop. arXiv:1812.03381
  - Uchendu et al., "Jump-Start Reinforcement Learning," ICML 2023. arXiv:2204.02372
  - Narvekar et al., "Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey," JMLR 2020. arXiv:2003.04960

### B.21 Residual learning on a base policy
- **Assumes:** A base controller (scripted, planner-based or BC) is roughly right, and what remains is a small, learnable correction.
- **Executable form:** `a = a_base(s) + clip(Δ_θ(s, a_base), −δ, δ)`, with Δ initialized to zero and the base frozen. Learn Δ by RL on task return, or by regression on correction data. Policy Decorator additionally bounds the residual and controls exploration.
- **Numpy-only feasible:** yes. Δ can be a linear, RBF or small-MLP model, optimized by CEM or ES on episodic success or return, or by ridge regression on corrections. The base can be the agent's own scripted skill.
- **Describes the skill to a caller as:** "Base = scripted controller X plus a learned residual bounded by δ (e.g., 5 mm per step). Behavior stays within δ of X, so X's description still applies."
- **Helps when / fails when:** Helps on precision and contact tasks (insertion, alignment) where the base gets close, and gives large sample-efficiency gains over learning from scratch. Fails when the base uses the wrong strategy, when δ is too small to fix the error, and when the base is out of its own distribution.
- **How to test it:** Base alone vs base + residual vs learning from scratch, at equal environment steps. Metrics: success rate and steps to reach it. Sweep δ.
- **References:**
  - Johannink et al., "Residual Reinforcement Learning for Robot Control," ICRA 2019. arXiv:1812.03201
  - Silver et al., "Residual Policy Learning," arXiv 2018. arXiv:1812.06298
  - Ankile et al., "From Imitation to Refinement — Residual RL for Precise Assembly," ICRA 2025. arXiv:2407.16677
  - Yuan et al., "Policy Decorator: Model-Agnostic Online Refinement for Large Policy Model," ICLR 2025. arXiv:2412.13630

### B.22 Regularization toward a prior / reference policy (BC terms, KL, demo replay)
- **Assumes:** A reference policy (BC from demos, or a previously learned skill) encodes good behavior. Improvement should trade reward against divergence from it instead of drifting freely.
- **Executable form:** Add one of these terms to the RL objective:
  - A BC term on demos with a decaying weight λ (DAPG).
  - `−Q(s, π(s)) + α||π(s) − a_demo||²` (TD3+BC).
  - `β·KL(π || π_prior)` (KL-regularized RL with behavior priors).
  - Demos kept in the replay buffer with 50/50 symmetric sampling (RLPD).

  For CEM/ES, penalize deviation from the BC action or from the BC-initialized parameters.
- **Numpy-only feasible:** yes. Each is a penalty term usable in any numpy optimizer, and the KL between Gaussian policies is closed form.
- **Describes the skill to a caller as:** "Fine-tuned from BC on N demos with deviation penalty λ. Demo-like except where reward improved; mean action deviation from the prior: x."
- **Helps when / fails when:** Helps with sparse rewards plus few demos, and prevents early catastrophic drift in fine-tuning. Fails when the prior is poor (the penalty holds the policy near bad behavior) or λ is too large (blocks improvement).
- **How to test it:** Sweep λ ∈ {0, small, large}. Metrics: success vs environment steps, and final mean action deviation from the prior.
- **References:**
  - Rajeswaran et al., "Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations" (DAPG), RSS 2018. arXiv:1709.10087
  - Fujimoto & Gu, "A Minimalist Approach to Offline Reinforcement Learning" (TD3+BC), NeurIPS 2021. arXiv:2106.06860
  - Tirumala et al., "Behavior Priors for Efficient Reinforcement Learning," JMLR 2022. arXiv:2010.14274
  - Ball et al., "Efficient Online Reinforcement Learning with Offline Data" (RLPD), ICML 2023. arXiv:2302.02948

---

# C. Architecture and model-structure priors

Scope: how the policy or model is *structured*: symmetry, object factorization, output distribution class, hierarchy and modularity, controller structure, pretrained components, world models, and spatial layout.

In the numpy-only sandbox, most of these priors cannot be network layers. They survive as **input/output parameterizations** (frames, sets, chunks, primitives, residuals), **explicit structure in code** (options, gates, relational features), or **test-time optimization** (argmin of an energy or cost). Each entry says which of these it becomes.

All references below were checked against arXiv, OpenAlex, Crossref, OpenReview or publisher pages (title, first author, year, venue). None are marked unverified.

---

### C.1 Group-equivariant policies (SO(2) / SE(2) / SE(3) / SIM(3))
- **Assumes:** Moving the whole scene by a group element g (rotation, translation, possibly scale) should move the correct action by the same g, so π(g·s) = g·π(s). Actions that break this for transformed scenes are ruled out by construction.
- **Executable form:** Equivariant layers inside the policy: steerable CNNs or equivariant point-cloud encoders feeding an equivariant head. For example, the EquiDiff denoiser is SO(2)-equivariant, EquiBot's encoder is SIM(3)-equivariant, and Equivariant Transporter makes its pick and place maps equivariant. Two cheaper routes give the same guarantee. (a) Canonicalize: map the input into a reference frame, run any model, then map the output back. (b) Symmetrize: average a black-box model over a finite group, π(s) = 1/|G| Σ_g g·f(g⁻¹·s). Augmenting the data with random g gives only a soft version of the prior.
- **Numpy-only feasible:** yes. Express positions and velocities relative to an anchor object or goal, and rotate them by −yaw of the anchor (SE(2)) or by R_anchorᵀ (SE(3)). Fit any regressor in that frame (kNN, ridge, GMR) and rotate the predicted action back. Symmetrizing over C_n rotations works around any numpy policy, but averaging a multimodal policy blurs its modes, so canonicalize instead when the policy is multimodal.
- **Describes the skill to a caller as:** "Equivariant to <group> about <anchor>. Works for any in-plane position and yaw of <object>. Tilt, scale and changes relative to the robot base are not covered."
- **Helps when / fails when:** Helps when the task physics really are invariant (tabletop push or pick-place with random yaw) and demos cover few orientations. Fails when the symmetry is broken by joint limits, reachability, table edges, walls or gravity (for SE(3) tilts), or when the anchor's orientation is ill-defined (symmetric objects).
- **How to test it:** Fit on the same demos with and without canonicalization. Test in new sim scenes with object yaw and position sampled outside the demo range, for example demos with yaw in ±30° and tests over the full circle. Report success rate per yaw bin and the equivariance error ‖π(g·s) − g·π(s)‖ for random g.
- **References:**
  - Wang, D., Walters, R., Platt, R. "SO(2)-Equivariant Reinforcement Learning." ICLR 2022. arXiv:2203.04439
  - Huang, H. et al. "Equivariant Transporter Network." RSS 2022. arXiv:2202.09400
  - Wang, D. et al. "Equivariant Diffusion Policy." CoRL 2024. arXiv:2407.01812 (reports a 21.9% average success gain over Diffusion Policy on 12 MimicGen tasks)
  - Yang, J. et al. "EquiBot: SIM(3)-Equivariant Diffusion Policy for Generalizable and Data Efficient Learning." CoRL 2024. arXiv:2407.01479

### C.2 Object-frame descriptors and task-frame selection (Neural Descriptor Fields, task-parameterized models)
- **Assumes:** A skill is defined by a relative pose between the end effector (or a held object) and a frame attached to a task-relevant object. That relative pose stays the same when the object moves and, within a category, when its shape varies.
- **Executable form:** NDF uses an SE(3)-equivariant descriptor field over query points. The demo gripper pose is encoded as the descriptors of points rigidly attached to the gripper, and at test time the gripper pose is optimized to match those descriptors. R-NDF applies the same idea to object–object relations such as mug-on-rack. Task-parameterized GMMs encode each demo in several candidate frames and combine the per-frame Gaussians by product at test time. Learned canonicalization predicts a canonical pose and then runs any backbone.
- **Numpy-only feasible:** yes. For each keyframe (grasp, pre-place, place), compute T_rel = T_frame⁻¹·T_ee for every candidate frame (each object, the goal, the world). Select the frame where T_rel varies least across demos; this selection is the learning step. Then execute T_ee* = T_frame,new · mean(T_rel), with rotations averaged via SVD or quaternions. For whole trajectories, express each segment in its selected frame, or blend frames by their precision (TP-GMM fitted with numpy EM).
- **Describes the skill to a caller as:** "Defined in <object>'s frame. Grasp and place poses move rigidly with <object>, so any pose of <object> works. Other object shapes work only if the frame lands on the same feature (handle, rim)."
- **Helps when / fails when:** Helps with few demos (5–10), large pose variation, and a clear attachment feature. Fails when the relevant frame switches mid-skill and the skill is not segmented there, when shape variation moves the feature relative to the object origin (use keypoints, see C.22), for symmetric objects, or with noisy pose estimates.
- **How to test it:** Compare world-frame replay, fixed object-frame replay and variance-selected frames on the same demos. Metrics: success rate and final relative-pose error in new scenes with randomized object poses. Log the across-demo variance of T_rel for each frame as the evidence for the selection.
- **References:**
  - Simeonov, A. et al. "Neural Descriptor Fields: SE(3)-Equivariant Object Representations for Manipulation." ICRA 2022. arXiv:2112.05124
  - Simeonov, A. et al. "SE(3)-Equivariant Relational Rearrangement with Neural Descriptor Fields." CoRL 2022. arXiv:2211.09786
  - Kaba, S.-O. et al. "Equivariance with Learned Canonicalization Functions." ICML 2023 (PMLR 202). arXiv:2211.06489
  - Calinon, S. "A tutorial on task-parameterized movement learning and retrieval." Intelligent Service Robotics 9(1):1–29, 2016. doi:10.1007/s11370-015-0187-9

### C.3 Permutation-invariant set encoders over objects (Deep Sets, Set Transformer, PointNet)
- **Assumes:** The scene is an unordered set of entities. The policy output must not depend on the order in which objects are listed, and the same per-object computation should apply to any number of objects.
- **Executable form:** f(X) = ρ(pool_i φ(x_i)) with sum, mean or max pooling (Deep Sets, PointNet), or self-attention blocks followed by attention pooling (Set Transformer). For object-centric policies, the per-entity features are object state concatenated with robot state and goal. Outputs are either pooled (invariant) or per-object (equivariant, e.g. "which object to act on").
- **Numpy-only feasible:** yes. Build hand-crafted per-object features φ (relative position to the gripper, distance to goal, flags), pool them with sum, max or softmin, and fit a small regressor ρ (ridge, kNN) on the pooled vector. The most useful numpy form is a per-object score followed by argmax, i.e. hard attention: choose the target object, then run a single-object policy in its frame (C.2).
- **Describes the skill to a caller as:** "Works for any number and order of <object type> instances. Handles one selected object at a time, chosen by <criterion>."
- **Helps when / fails when:** Helps when the object count varies between training and test, or objects are interchangeable. Fails when object identity or order carries meaning (a stacking order, a colour-specific goal) and is not encoded as a feature, or when sum pooling saturates as the count grows (use mean or max).
- **How to test it:** Train on scenes with N objects and test with N ± k objects and shuffled object order. Compare against a fixed-order concatenated-vector policy. Metrics: success rate as a function of object count, and output change under permutation (should be 0).
- **References:**
  - Zaheer, M. et al. "Deep Sets." NeurIPS 2017. arXiv:1703.06114
  - Lee, J. et al. "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks." ICML 2019. arXiv:1810.00825
  - Qi, C. R. et al. "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation." CVPR 2017. arXiv:1612.00593
  - Zhou, A. et al. "Policy Architectures for Compositional Generalization in Control." RLC 2024 (RLJ); arXiv:2203.05960 (Deep Sets and self-attention policies extrapolate to unseen numbers of entities)

### C.4 Relational / graph-structured policies (graph networks, relational RL)
- **Assumes:** What matters is the pairwise relations between entities (contact, on-top-of, distance, alignment), and the same relation function applies to every pair. Knowledge learned on small scenes transfers to larger ones.
- **Executable form:** A graph with objects, robot and goals as nodes and relations as edges. Message passing computes e_ij = φ_e(x_i, x_j), then n_i = φ_n(x_i, Σ_j e_ij), then the readout. Relational RL applies self-attention across entities. GNN relational classifiers predict how the relations change after an action and support planning over them.
- **Numpy-only feasible:** yes. Compute a pairwise feature tensor (N×N×d: relative positions, distances, contact flags) and hand-written predicates such as on(i,j), near(i,j) and aligned(i,j). The policy or termination logic can operate on these predicates, and one or two rounds of sum-aggregation with fixed or ridge-fitted weights give a numpy "GNN". The relational predicates also give preconditions and effects for the caller.
- **Describes the skill to a caller as:** "Precondition: <relation set>. Effect: <relation changes>. Independent of how many other objects are present, as long as they are not in relation R with the target."
- **Helps when / fails when:** Helps in multi-object tasks (stacking, sorting, rearrangement) and with curricula that grow the object count. Fails when the relevant interaction is higher-order or global (a whole pile's stability), or when the predicates are mis-specified (thresholds tuned on the demos).
- **How to test it:** Compare a relational-feature policy with a flat-state policy on the same demos. Test on scenes with more objects, or different pairings, than in training. Metrics: success rate, and accuracy of predicted relation changes against the simulator's outcome.
- **References:**
  - Battaglia, P. W. et al. "Relational inductive biases, deep learning, and graph networks." arXiv:1806.01261, 2018
  - Zambaldi, V. et al. "Deep reinforcement learning with relational inductive biases." ICLR 2019 (arXiv:1806.01830, titled "Relational Deep Reinforcement Learning")
  - Li, R. et al. "Towards Practical Multi-Object Manipulation using Relational Reinforcement Learning." ICRA 2020. arXiv:1912.11032 (graph-based policy plus curriculum learns to stack 6 blocks)
  - Huang, Y. et al. "Planning for Multi-Object Manipulation with Graph Neural Network Relational Classifiers." ICRA 2023. arXiv:2209.11943

### C.5 Object slots and object-token attention (slot attention, object-centric transformers)
- **Assumes:** A scene is best represented as a fixed-width token per object. The policy should attend to a small subset of task-relevant objects and ignore distractors.
- **Executable form:** From pixels, slot attention (iterative competitive attention) or object proposals produce object tokens. A transformer over the object tokens plus robot and goal tokens then outputs actions (VIOLA, VIMA). The Entity Interaction Transformer in EIT handles multi-object goals with ordering dependencies.
- **Numpy-only feasible:** yes (trivially for the representation). The sandbox state is already object-centric, so slot discovery is unnecessary. What remains is to (a) keep a consistent per-object token array with stable identities and (b) apply attention as a softmax over per-object relevance scores, softmax(q·k_i/τ), where q encodes the skill goal and k_i the object features. The attention weights double as an explanation of which objects the skill uses.
- **Describes the skill to a caller as:** "Reads only <objects with nonzero attention>. Robust to added distractor objects of other types."
- **Helps when / fails when:** Helps in cluttered scenes with distractors or when goals name specific objects. Fails when the relevant object cannot be distinguished by its features (two identical objects with different roles), or when an interaction involves a non-attended object (an obstacle the skill ignores).
- **How to test it:** Add 0, 2 or 5 distractor objects in new scenes and compare an attention/selection policy against a flat-state policy. Metrics: success rate as distractors increase, and whether the attention mass falls on the ground-truth relevant objects.
- **References:**
  - Locatello, F. et al. "Object-Centric Learning with Slot Attention." NeurIPS 2020. arXiv:2006.15055
  - Zhu, Y. et al. "VIOLA: Imitation Learning for Vision-Based Manipulation with Object Proposal Priors." CoRL 2022. arXiv:2210.11339
  - Jiang, Y. et al. "VIMA: General Robot Manipulation with Multimodal Prompts." ICML 2023. arXiv:2210.03094
  - Haramati, D., Daniel, T., Tamar, A. "Entity-Centric Reinforcement Learning for Object Manipulation from Pixels." ICLR 2024. arXiv:2404.01220 (trained with 3 objects, generalizes to more than 10)

### C.6 Diffusion policies
- **Assumes:** The demo action distribution is multimodal and high-dimensional (whole action sequences). A policy should sample one coherent mode rather than regress to the mean. Predicting a short action sequence and executing it with receding horizon gives temporal consistency.
- **Executable form:** A conditional denoiser ε_θ(a_t, t | obs) is trained on noised demo action chunks. At run time, iterative denoising (DDPM/DDIM, K steps) produces a T_p-step action chunk; the first T_a steps are executed and the rest re-planned. Diffuser diffuses whole state-action trajectories and adds guidance toward reward or goal. DP3 conditions on a compact point-cloud embedding.
- **Numpy-only feasible:** partly. Training a real denoiser needs torch. A numpy stand-in that keeps the prior (multimodal, sample-then-commit, chunked) is nonparametric. Retrieve the k nearest demo states in canonical features, sample one neighbour's action chunk (this picks a mode), and optionally refine it by gradient steps on the closed-form log-density of a Gaussian KDE over the neighbours' chunks. It generalizes only as far as the features do.
- **Describes the skill to a caller as:** "Stochastic and multimodal: may choose different valid strategies (e.g. go left or right around an obstacle) on different runs. Commits to a strategy for T_a steps."
- **Helps when / fails when:** Helps with multimodal human demos and long-horizon smooth motions. Fails when inference latency matters (many denoising steps), with very few demos (it memorizes), or when the task needs a precise point estimate that the sampling noise perturbs.
- **How to test it:** Compare against a unimodal Gaussian or MSE regressor on the same demos. Metrics: success rate in new scenes, and a mode-averaging check: the fraction of rollouts whose path passes between two demo modes (e.g. into the obstacle).
- **References:**
  - Chi, C. et al. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." RSS 2023 (extended version on arXiv). arXiv:2303.04137
  - Janner, M. et al. "Planning with Diffusion for Flexible Behavior Synthesis." ICML 2022. arXiv:2205.09991
  - Ze, Y. et al. "3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations." RSS 2024. arXiv:2403.03954

### C.7 Flow-matching policies
- **Assumes:** Same multimodality prior as diffusion, but the policy is a learned velocity field that transports noise to actions along near-straight paths. This makes sampling cheap (few integration steps) and training a simple regression.
- **Executable form:** Train v_θ(x_t, t | obs) to regress (a − x0), where x_t = (1−t)·x0 + t·a and x0 is noise. At run time, integrate dx/dt = v_θ from t=0 to 1 in a few Euler steps. π0 attaches a flow-matching action expert to a VLM and outputs action chunks. Riemannian FM keeps outputs on manifolds such as SO(3) or the sphere. Consistency variants (FlowPolicy) aim at one-step sampling.
- **Numpy-only feasible:** partly. For a finite demo set, the optimal velocity field is analytic: v(x,t) = Σ_i w_i(x,t)·(a_i − x)/(1−t), with w_i ∝ k(obs, obs_i)·N(x; t·a_i, (1−t)²I). This is exactly implementable in numpy, and a state kernel k supplies the conditioning. It reproduces demo modes (memorization); generalization comes only from canonicalized features in the kernel. A learned field needs torch.
- **Describes the skill to a caller as:** "Multimodal sampler over <action-chunk> with N integration steps. Fast enough for <Hz>. Output stays on <manifold> if Riemannian."
- **Helps when / fails when:** Helps with multimodal demos when inference must be fast, and with outputs on non-Euclidean manifolds (orientations). Fails in the same ways as diffusion with few demos (memorization), and when too few integration steps cut corners between modes.
- **How to test it:** Sweep the number of integration steps (1, 2, 5, 10) and compare with a diffusion or kNN-sampling baseline. Metrics: success rate, wall-clock per action, and the fraction of samples landing between modes.
- **References:**
  - Lipman, Y. et al. "Flow Matching for Generative Modeling." ICLR 2023. arXiv:2210.02747
  - Black, K. et al. "π0: A Vision-Language-Action Flow Model for General Robot Control." RSS 2025. arXiv:2410.24164
  - Braun, M. et al. "Riemannian Flow Matching Policy for Robot Motion Learning." IROS 2024. arXiv:2403.10672
  - Zhang, Q. et al. "FlowPolicy: Enabling Fast and Robust 3D Flow-based Policy via Consistency Flow Matching for Robot Manipulation." AAAI 2025. arXiv:2412.04987

### C.8 Action chunking with a latent style variable (ACT / CVAE)
- **Assumes:** Predicting k future actions at once lowers the effective horizon and reduces compounding error. Demo idiosyncrasies (pauses, style) are absorbed by a latent variable rather than averaged away.
- **Executable form:** A CVAE whose encoder maps the demo chunk to z and whose transformer decoder maps (obs, z) to k actions; at test time z = 0 (the prior mean). Execution uses temporal ensembling: every step, average all overlapping chunk predictions for the current time with weights exp(−m·i). Later work addresses the loss of reactivity within a chunk by sampling several chunks and choosing the one consistent with the previous chunk and with a stronger or weaker model (bidirectional decoding), or by inpainting the next chunk while the current one executes (real-time chunking).
- **Numpy-only feasible:** yes, for chunking and ensembling. Chunks can come from a nearest-neighbour or ridge model that predicts k·d_a outputs, with temporal ensembling applied at run time. Replace the CVAE latent with a discrete mode: k-means (scipy.cluster.vq) over demo chunks, choose the cluster at chunk start, then regress within the cluster.
- **Describes the skill to a caller as:** "Open-loop for up to k steps (≈ k·dt seconds) between replans. Reacts to disturbances within about one chunk. Smooth because of ensembling."
- **Helps when / fails when:** Helps with fine manipulation from human teleop demos and with non-Markovian pauses. Fails with fast disturbances or moving targets inside a chunk; long chunks lower reactivity, and ensembling can blur between modes.
- **How to test it:** Sweep the chunk length k ∈ {1, 5, 20, 50} with and without temporal ensembling. Metrics: success rate in nominal new scenes and in scenes with a mid-episode object perturbation. This measures the precision versus reactivity trade-off.
- **References:**
  - Zhao, T. Z. et al. "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware." RSS 2023. arXiv:2304.13705
  - Sohn, K., Lee, H., Yan, X. "Learning Structured Output Representation using Deep Conditional Generative Models." NeurIPS 2015
  - Liu, Y. et al. "Bidirectional Decoding: Improving Action Chunking via Guided Test-Time Sampling." ICLR 2025. arXiv:2408.17355
  - Black, K. et al. "Real-Time Execution of Action Chunking Flow Policies." NeurIPS 2025. arXiv:2506.07339

### C.9 Mixture-density and discretized multimodal action heads (MDN, GMR, BeT, VQ-BeT)
- **Assumes:** p(a|s) is a small number of modes, each roughly Gaussian or a discrete bin plus a small continuous offset. Explicit mixture weights make the modes inspectable.
- **Executable form:** An MDN outputs mixture weights, means and covariances; sampling picks a component, then samples within it. Gaussian mixture regression fits a joint GMM over (s, a) and conditions it on s. BeT clusters actions with k-means into bins, then predicts a bin class plus a residual offset. VQ-BeT replaces k-means with residual vector quantization of action chunks. RT-style heads discretize each action dimension into bins.
- **Numpy-only feasible:** yes. Fit a joint GMM on [s; a] with numpy EM and condition analytically (GMR), which gives multimodal p(a|s) in closed form. Alternatively, run k-means on actions or chunks (scipy.cluster.vq.kmeans2), train a classifier from s to bin (softmax regression, or kNN votes), and fit a ridge offset per bin.
- **Describes the skill to a caller as:** "Has M discrete strategies (listed with their mixture weight per context). Picks one per episode or chunk. Within a strategy, the action is <mean ± std>."
- **Helps when / fails when:** Helps with a few clean modes, low-dimensional actions, and when a caller benefits from named strategies. Fails with high-dimensional or long action chunks (the component count explodes), with MDN numerical instability or mode collapse, and when the bin resolution is too coarse for precision tasks without offsets.
- **How to test it:** Compare 1 component (plain regression) with M components, choosing M by BIC or held-out likelihood on demos. Metrics: success rate in new scenes and log-likelihood of held-out demo actions (cross-validated across demos, not held-out episodes at evaluation).
- **References:**
  - Bishop, C. M. "Mixture Density Networks." Tech. Rep. NCRG/94/004, Aston University, 1994
  - Calinon, S., Guenter, F., Billard, A. "On Learning, Representing, and Generalizing a Task in a Humanoid Robot." IEEE Trans. SMC-B 37(2):286–298, 2007 (GMM/GMR)
  - Shafiullah, N. M. M. et al. "Behavior Transformers: Cloning k modes with one stone." NeurIPS 2022. arXiv:2206.11251
  - Lee, S. et al. "Behavior Generation with Latent Actions." ICML 2024 (PMLR 235). arXiv:2403.03181

### C.10 Energy-based / implicit policies
- **Assumes:** A good action is easier to recognize than to generate. The policy is a = argmin_a E(s, a), which can represent discontinuous and multi-valued mappings that explicit regressors smooth over.
- **Executable form:** Learn E_θ(s, a) contrastively, pushing demo actions low and sampled counter-examples high (InfoNCE; ranking NCE with learned negative samplers). Inference uses derivative-free optimization (iterative sampling or CEM) or Langevin dynamics over a. Soft Q-learning is the RL counterpart, with π ∝ exp(Q/α).
- **Numpy-only feasible:** yes. Define E as (i) a hand-built cost (distance to a goal relation, collision penalty), (ii) the negative log of a KDE over demo (s, a) pairs in canonical features, or (iii) a ridge-fitted quadratic in features. Minimize it at each step with CEM, or with scipy.optimize over a (or a short action sequence; see C.16).
- **Describes the skill to a caller as:** "Chooses each action by optimizing <energy terms>. Handles sharp decision boundaries (which side to push from). Per-step cost is <n samples × m iterations>."
- **Helps when / fails when:** Helps when the mapping is discontinuous (contact switches, push-side selection), when constraints are easy to write as costs, and in low-dimensional action spaces. Fails in high-dimensional actions (sampling cost grows quickly with dimension), when training is unstable without good negatives, and when inference-time optimization is too slow for the control rate.
- **How to test it:** On the same demos, compare an explicit regressor (MSE) with argmin of the energy. Evaluate in new scenes placed near the decision boundaries (e.g. object positions where the correct push side flips). Metrics: success rate, and the error of the chosen action near the discontinuity.
- **References:**
  - Florence, P. et al. "Implicit Behavioral Cloning." CoRL 2021 (PMLR 164). arXiv:2109.00137
  - Haarnoja, T. et al. "Reinforcement Learning with Deep Energy-Based Policies." ICML 2017. arXiv:1702.08165
  - Singh, S., Tu, S., Sindhwani, V. "Revisiting Energy Based Models as Policies: Ranking Noise Contrastive Estimation and Interpolating Energy Models." TMLR 2024. arXiv:2309.05803

### C.11 Hierarchical policies and options
- **Assumes:** Behaviour decomposes in time into reusable sub-behaviours, each with its own start condition and stopping condition. A high-level policy chooses among them at a coarser time scale.
- **Executable form:** An option is a tuple (initiation set I, intra-option policy π_o, termination β_o). The high-level policy picks o at each termination. Variants: option-critic learns all three parts end-to-end; HIRO has the high level emit a subgoal state offset every c steps for a goal-conditioned low level; Hi Robot has a VLM emit language subcommands to a VLA.
- **Numpy-only feasible:** yes. Options are plain Python objects with `can_start(s)`, `act(s)` and `done(s)` predicates written or fitted from demo segments (segment at gripper open/close, contact changes or velocity zeros). The high level can be a finite-state machine, a planner over option pre- and postconditions, or a lookup learned from demo segment order. Each option can use any other prior in this catalog.
- **Describes the skill to a caller as:** "Composite skill: sequence or graph of sub-skills [reach → grasp → lift → place]. Each has a start condition and a termination condition. Fails over to <recovery option> when <condition>."
- **Helps when / fails when:** Helps with long-horizon tasks, sparse rewards, and reusing sub-skills across tasks. Fails when segment boundaries are wrong (termination fires early or late), when sub-skill end states fall outside the next skill's initiation set (hand-off gaps), or when the hierarchy is imposed on a task that is really one continuous motion.
- **How to test it:** Compare a flat policy with the option hierarchy on the same demos. Metrics: end-to-end success in new scenes, and per-option success including hand-off success (end state of option k ∈ I of option k+1). Count failures by option.
- **References:**
  - Sutton, R. S., Precup, D., Singh, S. "Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning." Artificial Intelligence 112(1–2):181–211, 1999
  - Bacon, P.-L., Harb, J., Precup, D. "The Option-Critic Architecture." AAAI 2017. arXiv:1609.05140
  - Nachum, O. et al. "Data-Efficient Hierarchical Reinforcement Learning." NeurIPS 2018. arXiv:1805.08296
  - Shi, L. X. et al. "Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models." ICML 2025. arXiv:2502.19417

### C.12 Latent skill spaces and skill-conditioned policies
- **Assumes:** Useful behaviours lie on a low-dimensional manifold of trajectory segments. One policy conditioned on a latent code z covers a family of related skills, and search or RL over z is easier than over raw actions.
- **Executable form:** Learn an encoder from trajectory segment to z and a decoder π(a | s, z). Sources of structure: play data (LfP), offline skill priors p(z|s) that guide RL (SPiRL), mutual-information skill discovery without reward (DIAYN), or quantized skill tokens (QueST). Downstream, a high-level policy or planner selects z.
- **Numpy-only feasible:** yes (linear version). Time-normalize demo segments in canonical frames, stack them, and use PCA/SVD for a linear skill basis (z = PCA coefficients). A Gaussian over demo z serves as the skill prior. Search z by CEM in the simulator under that prior, or regress z from context with ridge. Nonlinear encoders need torch.
- **Describes the skill to a caller as:** "Parameterized skill family with a d-dimensional parameter z. Valid z lies within the demo-derived prior (radius r). Here is how z maps to observable effects (e.g. z₁ ≈ reach height)."
- **Helps when / fails when:** Helps when there are many related demos, a downstream search needs a compact space, and adaptation must use few rollouts. Fails when the demos do not span the needed variation (the right skill is outside the manifold), when z is entangled and uninterpretable for the caller, or when there are too few demos for the latent dimensionality.
- **How to test it:** Adapt to new sim scenes by (a) CEM over z inside the prior and (b) CEM over raw action or waypoint parameters, with an equal rollout budget. Metrics: success versus number of sim rollouts, and reconstruction error of held-in demos as a function of d.
- **References:**
  - Eysenbach, B. et al. "Diversity is All You Need: Learning Skills without a Reward Function." ICLR 2019. arXiv:1802.06070
  - Lynch, C. et al. "Learning Latent Plans from Play." CoRL 2019. arXiv:1903.01973
  - Pertsch, K., Lee, Y., Lim, J. J. "Accelerating Reinforcement Learning with Learned Skill Priors." CoRL 2020. arXiv:2010.11944
  - Mete, A. et al. "QueST: Self-Supervised Skill Abstractions for Learning Continuous Control." NeurIPS 2024. arXiv:2407.15840

### C.13 Mixture of experts and modular policies
- **Assumes:** The state space splits into regimes (contact modes, task variants, robot bodies), each simple enough for its own expert. A gate routes each input to one or a few experts, and modules can be recombined.
- **Executable form:** π(a|s) = Σ_k g_k(s)·π_k(a|s) with a softmax or sparse top-k gate. Modular networks compose robot modules with task modules (Devin et al.). Sparse Diffusion Policy activates task-specific experts inside a diffusion transformer. Diffusion policies can also be distilled into a Gaussian MoE for fast, inspectable inference.
- **Numpy-only feasible:** yes. The gate is nearest-centroid, logistic regression (numpy gradient descent) or a hand-coded mode detector (in contact, grasped, near goal). Experts are ridge or linear-feedback controllers per region, which gives a piecewise-affine policy. Fit by EM (alternate gating assignment and expert refit), or segment demos by contact events and fit one expert per segment.
- **Describes the skill to a caller as:** "Has K regimes: <names>, with the switching condition for each. Behaviour inside each regime is <simple law>. Regimes the gate never saw in demos are unsupported."
- **Helps when / fails when:** Helps with hybrid dynamics (free space versus contact), multi-task data, and interpretability. Fails when gate errors near regime boundaries cause chattering, when an expert is starved of data in a rare mode, or when K is too large for the demo count.
- **How to test it:** Sweep K ∈ {1, 2, 4} against a single global regressor. Metrics: success in new scenes, the number of gate switches per episode (chattering), and gate agreement with ground-truth contact mode from the simulator.
- **References:**
  - Jacobs, R. A., Jordan, M. I., Nowlan, S. J., Hinton, G. E. "Adaptive Mixtures of Local Experts." Neural Computation 3(1):79–87, 1991
  - Devin, C. et al. "Learning Modular Neural Network Policies for Multi-Task and Multi-Robot Transfer." ICRA 2017. arXiv:1609.07088
  - Wang, Y. et al. "Sparse Diffusion Policy: A Sparse, Reusable, and Flexible Policy for Robot Learning." CoRL 2024. arXiv:2407.01531
  - Zhou, H. et al. "Variational Distillation of Diffusion Policies into Mixture of Experts." NeurIPS 2024. arXiv:2406.12538

### C.14 Residual policies on top of a base controller
- **Assumes:** An existing controller (scripted, model-based, or a BC policy) is mostly right. The learned part only needs a small correction, which is easier to learn and safer to explore than a full policy.
- **Executable form:** a = a_base(s) + clip(r_θ(s, a_base), −δ, δ). r_θ is learned with RL (initialized near zero) or regressed on (a_demo − a_base). Recent work trains per-step closed-loop residuals on top of chunked BC policies (Ankile et al.), or bounds the residual and controls exploration on top of large pretrained policies (Policy Decorator).
- **Numpy-only feasible:** yes. The base is a scripted P/PD controller toward a C.2 target. The residual is ridge regression on demo (a − a_base) over canonical features, or a few parameters (offsets, gains) tuned by CEM or random search in the simulator. Keep δ small and report it.
- **Describes the skill to a caller as:** "Base behaviour: <controller> toward <target>. Learned correction of at most δ per step. If the correction is disabled, the skill still does <base behaviour>."
- **Helps when / fails when:** Helps when a reasonable controller exists and failures are small systematic errors (friction, offsets, precision insertion), and when data or rollouts are scarce. Fails when the base is qualitatively wrong (a different strategy is needed), when δ is too small to fix it, or when the residual learns to cancel the base.
- **How to test it:** Evaluate base only, residual only (the full policy without the base), and base + residual in new scenes. Metrics: success rate and final-error distribution. Also log the residual magnitude, which should be small relative to a_base.
- **References:**
  - Silver, T. et al. "Residual Policy Learning." arXiv:1812.06298, 2018
  - Johannink, T. et al. "Residual Reinforcement Learning for Robot Control." ICRA 2019. arXiv:1812.03201
  - Ankile, L. et al. "From Imitation to Refinement — Residual RL for Precise Assembly." arXiv:2407.16677, 2024
  - Yuan, X. et al. "Policy Decorator: Model-Agnostic Online Refinement for Large Policy Model." ICLR 2025. arXiv:2412.13630

### C.15 Controller-structured action spaces (impedance / operational-space outputs, movement-primitive layers)
- **Assumes:** The right thing to learn is the parameters of a well-understood controller (setpoints, stiffness, damping, attractor goals), not raw torques. The controller supplies stability, smoothness and compliance.
- **Executable form:** The policy outputs an end-effector pose delta plus impedance gains, which an operational-space controller converts to torques (VICES). Alternatively, the network outputs DMP parameters (goal, forcing weights) that a differentiable DMP integrates (Neural Dynamic Policies), or the policy predicts target stiffness alongside its trajectory (adaptive compliance). The choice of action space (joint versus Cartesian, delta versus absolute) is itself a prior with large effects.
- **Numpy-only feasible:** yes. Implement a DMP or PD/impedance law in numpy and fit its parameters: DMP forcing weights by least squares on the demos; the goal from the C.2 frame; gains by CEM in the simulator. Choose the action space on simulator evidence rather than by assumption.
- **Describes the skill to a caller as:** "Outputs <setpoint + stiffness>. Converges to goal g from any start (attractor). Compliant along <axes> with stiffness K. Speed can be scaled by τ."
- **Helps when / fails when:** Helps in contact-rich tasks (wiping, insertion, door opening), sim-to-real transfer, and when smoothness or safety matters. Fails when the task needs behaviour outside the controller family (impulsive motions, deliberate collisions), when the controller model is mismatched (the simulator does not implement OSC), or when many gains are poorly identified from few demos.
- **How to test it:** Fit the same demos with raw-action output and with controller-parameter output. Test in new scenes, including perturbed start states and contact-geometry variation. Metrics: success, contact-force peaks or energy, and success under start-state perturbation.
- **References:**
  - Martín-Martín, R. et al. "Variable Impedance Control in End-Effector Space: An Action Space for Reinforcement Learning in Contact-Rich Tasks." IROS 2019. arXiv:1906.08880
  - Ijspeert, A. J. et al. "Dynamical Movement Primitives: Learning Attractor Models for Motor Behaviors." Neural Computation 25(2):328–373, 2013
  - Bahl, S. et al. "Neural Dynamic Policies for End-to-End Sensorimotor Learning." NeurIPS 2020. arXiv:2012.02788
  - Aljalbout, E. et al. "On the Role of the Action Space in Robot Manipulation Learning and Sim-to-Real Transfer." IEEE RA-L 2024. arXiv:2312.03673

### C.16 Optimization as the policy (differentiable MPC, trajectory-optimization and planning layers)
- **Assumes:** Good behaviour is the solution of a short-horizon optimal-control problem. What varies across tasks and scenes is the cost and dynamics parameters, so learning should target those and leave the optimizer fixed.
- **Executable form:** The policy is a = first action of argmin_{a_{0:H}} Σ c_θ(x_t, a_t) subject to x_{t+1} = f_θ(x_t, a_t), solved by iLQR, MPPI or CEM. θ is learned end-to-end by differentiating through the solver (Differentiable MPC; DiffTORI) or by RL with the MPC as actor (AC-MPC). Value Iteration Networks embed a planner as a differentiable layer.
- **Numpy-only feasible:** yes. Solve with scipy.optimize (SLSQP or L-BFGS-B over an action sequence) or with numpy CEM/MPPI, using the simulator or a learned model (C.20) as f. Learn cost weights by inverse optimization: choose θ so the demo trajectories score better than perturbed ones (a margin or ranking loss, fitted with scipy). The fitted cost terms are the description of the skill.
- **Describes the skill to a caller as:** "Minimizes w₁·‖goal error‖ + w₂·effort + w₃·constraint penalties over an H-step horizon. Re-plans every step, so it adapts to disturbances within the model's validity."
- **Helps when / fails when:** Helps when the objective is easy to state but the motion is not, with constraints (obstacles, joint limits), and when disturbances call for re-planning. Fails with model error on long horizons, with non-smooth contact dynamics that break gradient solvers (use sampling), when compute exceeds the control budget, and when the cost is mis-specified so the optimizer exploits it.
- **How to test it:** Compare direct regression of actions with MPC using the learned cost, under equal demo data. Evaluate in new scenes and with mid-episode disturbances. Metrics: success, solve time per step, and how often solutions violate constraints the simulator enforces.
- **References:**
  - Amos, B. et al. "Differentiable MPC for End-to-end Planning and Control." NeurIPS 2018. arXiv:1810.13400
  - Tamar, A. et al. "Value Iteration Networks." NeurIPS 2016. arXiv:1602.02867
  - Wan, W. et al. "DiffTORI: Differentiable Trajectory Optimization for Deep Reinforcement and Imitation Learning." NeurIPS 2024. arXiv:2402.05421
  - Romero, A. et al. "Actor-Critic Model Predictive Control: Differentiable Optimization meets Reinforcement Learning for Agile Flight." IEEE T-RO 2025. arXiv:2306.09852

### C.17 Pretrained perception representations (R3M, MVP, VC-1, DINO-style features)
- **Assumes:** Visual features pretrained on large human-video or image corpora (time-contrastive, masked autoencoding, self-distillation) already encode what matters for manipulation. A small policy on frozen features therefore needs fewer demos.
- **Executable form:** Use a frozen or lightly fine-tuned encoder (ResNet or ViT) on each observation and a small MLP, diffusion or transformer head on the embedding. VC-1 benchmarks several such encoders across embodied tasks.
- **Numpy-only feasible:** no (needs images and torch). In the sandbox, the given object-centric state plays this role; the prior reduces to "use the provided features rather than learning a representation". The closest numpy analogue is reusing a feature map fitted on one skill's data (a PCA basis, canonical-frame features) for another skill.
- **Describes the skill to a caller as:** "Perception inherits the pretraining distribution of <encoder>. Robust to <lighting/texture> variation seen in pretraining. Not tuned to this scene."
- **Helps when / fails when:** Helps with pixel inputs and few in-domain demos when the visual domain overlaps the pretraining data. Fails when the pretraining domain mismatches (sim renders, unusual viewpoints), and a well-augmented from-scratch encoder can match frozen features; no single pretrained encoder dominates all tasks.
- **How to test it:** Compare a frozen pretrained encoder, a from-scratch encoder with augmentation, and ground-truth state, all with the same head and demos. Metric: success rate against the number of demos. For this sandbox, record the test as not applicable.
- **References:**
  - Nair, S. et al. "R3M: A Universal Visual Representation for Robot Manipulation." CoRL 2022. arXiv:2203.12601
  - Xiao, T. et al. "Masked Visual Pre-training for Motor Control." arXiv:2203.06173, 2022 (MVP)
  - Majumdar, A. et al. "Where are we in the search for an Artificial Visual Cortex for Embodied Intelligence?" NeurIPS 2023. arXiv:2303.18240 (VC-1; no single representation wins everywhere)
  - Hansen, N. et al. "On Pre-Training for Visuo-Motor Control: Revisiting a Learning-from-Scratch Baseline." ICML 2023. arXiv:2212.05749 (negative result: from-scratch with augmentation is competitive)

### C.18 Vision-language-action models (VLAs) as the policy backbone
- **Assumes:** A vision-language model pretrained on web data and then co-trained or fine-tuned on large multi-robot datasets carries semantic and physical priors. Conditioning it on language gives broad zero-shot or few-shot generalization.
- **Executable form:** Actions are emitted as discretized tokens from the language model (RT-2, OpenVLA; OpenVLA fuses DINOv2 and SigLIP visual features), or by a continuous action head such as diffusion (Octo) or a flow-matching action expert (π0) producing action chunks. Adaptation to a new robot or task is by fine-tuning (full or LoRA).
- **Numpy-only feasible:** no (needs images, GPUs and torch). The sandbox-relevant lessons are architectural: language-level task conditioning, action chunking (C.8), and separating a slow reasoning module from a fast action module (C.11).
- **Describes the skill to a caller as:** "Language-conditioned generalist. Competence comes from the pretraining and fine-tuning distribution, with no explicit guarantee on any particular object or pose. Latency <x> ms per chunk."
- **Helps when / fails when:** Helps with open-vocabulary instructions, semantic generalization (unseen objects or categories), and real-world visual diversity. Fails on precise, contact-rich or high-frequency control, on embodiments or action spaces far from the training mix, and with low compute budgets. Its failures are hard to predict or explain.
- **How to test it:** Compare fine-tuning the VLA with training a from-scratch specialist on the same demos. Metrics: success on in-distribution and semantically novel (unseen object) scenes, and latency. Not runnable in this sandbox.
- **References:**
  - Brohan, A. et al. "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control." CoRL 2023 (proceedings list Zitkovich, B. first). arXiv:2307.15818
  - Kim, M. J. et al. "OpenVLA: An Open-Source Vision-Language-Action Model." CoRL 2024. arXiv:2406.09246
  - Octo Model Team (Ghosh, D. et al.). "Octo: An Open-Source Generalist Robot Policy." RSS 2024. arXiv:2405.12213
  - Black, K. et al. "π0: A Vision-Language-Action Flow Model for General Robot Control." RSS 2025. arXiv:2410.24164

### C.19 LLM/VLM planners and code generation as architecture components
- **Assumes:** A language model's commonsense and coding ability can do the high-level decomposition, parameter choice and constraint specification, provided each proposal is grounded by feasibility or value checks and by low-level primitives or optimizers.
- **Executable form:** Four patterns. SayCan scores each skill by LLM usefulness × a learned affordance value. Code as Policies has the LLM write a program that calls perception and control APIs. VoxPoser composes 3D value and cost maps that a motion planner optimizes. ReKep has a VLM write keypoint-constraint functions that a solver optimizes.
- **Numpy-only feasible:** yes, and this is the sandbox's native form: the coding agent is the LLM component. Its outputs are (a) a program over parameterized primitives, (b) cost or constraint functions over object state for scipy to optimize (C.16), and (c) a plan over the skill library, checked for feasibility by rollouts in the simulator (a SayCan-style affordance equal to empirical success rate). The runtime policy is plain numpy code.
- **Describes the skill to a caller as:** "The skill is a program: <primitive calls>, with parameters from <rule/optimizer>. Valid under <assumptions written in the code>. Grounding checked by N simulator rollouts."
- **Helps when / fails when:** Helps with long-horizon, compositional or semantically specified tasks, and when a caller must read and edit the skill. Fails when the LLM hallucinates API semantics or physical feasibility (plans that are not grounded), when geometric precision is required beyond what is reasoned in text, and when unverified proposals are accepted.
- **How to test it:** Compare an LLM-proposed plan or code executed directly with the same proposal filtered by simulator feasibility checks, and with a hand-written script. Metrics: success in new scenes, and the fraction of proposals rejected by the grounding check.
- **References:**
  - Ahn, M. et al. "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances." CoRL 2022 (PMLR 205 lists Ichter, B. first). arXiv:2204.01691
  - Liang, J. et al. "Code as Policies: Language Model Programs for Embodied Control." ICRA 2023. arXiv:2209.07753
  - Huang, W. et al. "VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models." CoRL 2023. arXiv:2307.05973
  - Huang, W. et al. "ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation." CoRL 2024. arXiv:2409.01652

### C.20 Learned world models and latent dynamics (PETS, Dreamer, TD-MPC, structured world models)
- **Assumes:** The environment's dynamics can be learned from interaction and then used to plan or to train a policy in imagination, which is more sample-efficient than model-free learning. A compact latent state (or a factorization into objects) makes this easier.
- **Executable form:** PETS uses an ensemble of probabilistic dynamics nets with CEM MPC. Dreamer learns a recurrent state-space model and trains an actor-critic on imagined rollouts. TD-MPC2 uses a decoder-free latent model with MPPI planning and a learned terminal value. C-SWM learns object-factored latent dynamics with a GNN transition model.
- **Numpy-only feasible:** partly. The sandbox simulator is already a perfect model, but it is reachable only through reset/step. If it cannot be set to an arbitrary mid-episode state, planning from the current state needs a learned model. numpy options: ridge regression on object-relative features, local-linear or kNN models, bootstrap ensembles for uncertainty (PETS-style), with CEM/MPPI planning (C.16). Use ensemble disagreement to flag out-of-model states.
- **Describes the skill to a caller as:** "Plans with a learned model valid near <data region>. Uncertainty is flagged when ensemble disagreement exceeds <threshold>. Replans every k steps."
- **Helps when / fails when:** Helps when rollouts are expensive, many tasks share one environment, and the dynamics are smooth. Fails when compounding model error over long horizons lets the planner exploit model errors, with discontinuous contact dynamics, and outside the training state distribution.
- **How to test it:** Compare MPC with the learned model, MPC with the true simulator (an oracle, if the state can be set), and a model-free policy at an equal interaction budget. Metrics: success, k-step prediction error against horizon, and the correlation between ensemble disagreement and failure.
- **References:**
  - Chua, K. et al. "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models." NeurIPS 2018. arXiv:1805.12114
  - Hafner, D. et al. "Mastering Diverse Domains through World Models." arXiv:2301.04104 (published as "Mastering diverse control tasks through world models," Nature 640:647–653, 2025)
  - Hansen, N., Su, H., Wang, X. "TD-MPC2: Scalable, Robust World Models for Continuous Control." ICLR 2024. arXiv:2310.16828
  - Kipf, T., van der Pol, E., Welling, M. "Contrastive Learning of Structured World Models." ICLR 2020. arXiv:1911.12247

### C.21 Physics-structured dynamics models (interaction networks, Lagrangian/Hamiltonian networks, sparse physical regression)
- **Assumes:** The dynamics obey known physical structure: energy conservation or Euler–Lagrange form, pairwise object interactions, or sparsity in a small library of physical terms. Building that structure in gives data efficiency and extrapolation.
- **Executable form:** Interaction networks apply a shared relation network per object pair and an object update. DeLaN parameterizes a positive-definite mass matrix (via a Cholesky factor) and a potential, then solves Euler–Lagrange. HNNs learn H and integrate symplectic gradients. SINDy fits dx/dt = Θ(x)·Ξ with sparse Ξ over a library Θ (polynomials, trig, friction terms).
- **Numpy-only feasible:** yes. SINDy is sequentially thresholded least squares in numpy. Parametric rigid-body or contact models (mass, friction coefficient, restitution) can be fitted with scipy.optimize.least_squares on simulator rollouts. A pairwise-interaction model sums a fitted per-pair force law over object pairs. Energy-conserving structure applies only to conservative subsystems.
- **Describes the skill to a caller as:** "Model: <explicit equations with fitted parameters (e.g. μ = 0.4)>. Extrapolates to unseen speeds and masses as long as <law> holds. Contact and friction handled by <term>."
- **Helps when / fails when:** Helps with few data points, extrapolation beyond the training range, and interpretable parameters to report. Fails when contact, friction and impacts violate the assumed structure (conservative HNNs cannot dissipate energy), and when the term library misses the true mechanism.
- **How to test it:** Fit a physics-structured model and a black-box model (kNN or ridge on raw state) on the same rollouts, then test on out-of-range velocities, masses or longer horizons. Metrics: multi-step prediction error, and success when each model is used inside MPC.
- **References:**
  - Battaglia, P. W. et al. "Interaction Networks for Learning about Objects, Relations and Physics." NeurIPS 2016. arXiv:1612.00222
  - Greydanus, S., Dzamba, M., Yosinski, J. "Hamiltonian Neural Networks." NeurIPS 2019. arXiv:1906.01563
  - Lutter, M., Peters, J. "Combining physics and deep learning to learn continuous-time dynamics models." IJRR 2023. arXiv:2110.01894 (journal treatment of Deep Lagrangian Networks; original: Lutter et al., ICLR 2019, arXiv:1907.04490)
  - Brunton, S. L., Proctor, J. L., Kutz, J. N. "Discovering governing equations from data by sparse identification of nonlinear dynamical systems." PNAS 113(15), 2016. doi:10.1073/pnas.1517384113

### C.22 Spatially grounded architectures (spatial-softmax keypoints, dense action maps, attention over 3D points)
- **Assumes:** Task-relevant information is where things are, and actions are best expressed as locations or poses in the same space as the observation. This gives translation equivariance and lets a policy score many candidate actions in parallel.
- **Executable form:** Spatial softmax turns conv feature maps into expected 2D keypoint coordinates that feed the controller. Fully convolutional action maps (Transporter) produce a dense pick heatmap and match cropped features by cross-correlation for placement. PerAct predicts the next keyframe as a voxel over a 3D grid. 3D Diffuser Actor uses relative 3D attention between scene points and denoised end-effector poses.
- **Numpy-only feasible:** yes (the state is already keypoints). The dense-map form translates directly: evaluate a numpy score over a grid of candidate targets (place positions, push contact points, approach angles) built from object-relative features and costs, then take the argmax or softmax-sample. This is a value map in the style of VoxPoser and Transporter. Attention over points becomes a distance-weighted aggregation over object points or keypoints.
- **Describes the skill to a caller as:** "Chooses <pick/place location> by scoring a grid over <workspace region> at resolution r. Top-k alternatives with their scores are available. Equivariant to translation (and to rotation if rotations are enumerated)."
- **Helps when / fails when:** Helps with pick-and-place and rearrangement, multimodal target choice (several valid slots), and few demos. Fails when resolution limits precision (grid too coarse), in high-dimensional or long-horizon action spaces that do not reduce to a spatial target, and with dynamic tasks where the trajectory matters more than the endpoint.
- **How to test it:** Compare a grid-scored target with a directly regressed target on the same demos. Test in new scenes with shifted or rotated layouts. Metrics: success rate, target placement error, and a resolution sweep (r = 2, 1, 0.5 cm).
- **References:**
  - Levine, S., Finn, C., Darrell, T., Abbeel, P. "End-to-End Training of Deep Visuomotor Policies." JMLR 17, 2016. arXiv:1504.00702 (spatial softmax)
  - Zeng, A. et al. "Transporter Networks: Rearranging the Visual World for Robotic Manipulation." CoRL 2020 (PMLR 155). arXiv:2010.14406
  - Shridhar, M., Manuelli, L., Fox, D. "Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation." CoRL 2022. arXiv:2209.05451
  - Ke, T.-W., Gkanatsios, N., Fragkiadaki, K. "3D Diffuser Actor: Policy Diffusion with 3D Scene Representations." CoRL 2024. arXiv:2402.10885

---

# D. Control, planning and structured-policy priors

Explicit algorithmic, symbolic and physical structure. Two sandbox facts recur below:
1. **No `set_state`.** A simulator rollout "from the current state" means `reset(seed)` followed by replaying the executed action prefix. That costs O(t) steps and is valid only if reset plus replay is deterministic. Check this once by replaying a demo and comparing states. Inner-loop planning therefore usually runs on a **numpy surrogate model** (analytic or fitted), and the real simulator is used to validate and to collect data.
2. **The coding agent plays the LLM/VLM role.** Foundation-model methods keep only their executable part: code, constraints, predicates and affordance functions written at skill-authoring time and computed from object-centric state.

All references were checked against arXiv/DataCite, Crossref, PMLR, NeurIPS or ICLR proceedings pages. Author lists are shortened to the first author.

---

### D.1 Inverse kinematics (damped least squares / multi-start)
- **Assumes:** Goals are best stated in task space (end-effector pose, keypoint position), and a kinematic map x = f(q) is known or locally linearizable. The desired joint solution is the one closest to the current or seed configuration.
- **Executable form:** Damped least squares iteration Δq = Jᵀ(JJᵀ + λ²I)⁻¹ e, with joint-limit clipping, a null-space term for secondary goals (posture, limit avoidance) and random restarts. TRAC-IK style: run several solvers or seeds and keep the first or lowest-cost solution.
- **Numpy-only feasible:** yes/partly. Not needed if actions are already Cartesian end-effector deltas. For joint actions, `scipy.optimize.least_squares` on a numpy forward-kinematics model works when the kinematic parameters are known. Without a kinematic model, fit a local Jacobian Δx ≈ J Δq by least squares from demo transitions or small probing actions (Broyden updates online). Finite-difference probing of the simulator is impossible without `set_state`.
- **Describes the skill to a caller as:** "reach(pose): returns a joint target; reports residual pose error, whether it hit joint limits, and the smallest singular value of J (distance to singularity)."
- **Helps when / fails when:** Helps when targets are poses or keypoints and the controller takes joint commands. / Fails near singularities and joint limits, with multimodal redundant solutions, or when the task is defined by force rather than pose.
- **How to test it:** In new scenes, compare IK-mapped Cartesian targets against joint actions regressed directly from demos. Metrics: final pose error and reach success, with failures binned by singular value. Ablate the damping λ and the number of restarts.
- **References:**
  - Wampler, C. W. (1986). Manipulator inverse kinematic solutions based on vector formulations and damped least-squares methods. IEEE Trans. Systems, Man, and Cybernetics 16(1). doi:10.1109/TSMC.1986.289285
  - Buss, S. R. (2004). Introduction to inverse kinematics with Jacobian transpose, pseudoinverse and damped least squares methods. Tutorial manuscript, UCSD. https://mathweb.ucsd.edu/~sbuss/ResearchWeb/ikmethods/index.html
  - Beeson, P. et al. (2015). TRAC-IK: An open-source library for improved solving of generic inverse kinematics. IEEE-RAS Humanoids 2015. doi:10.1109/HUMANOIDS.2015.7363472
  - Ames, B. et al. (2022). IKFlow: Generating diverse inverse kinematics solutions. IEEE RA-L. arXiv:2111.08933, doi:10.1109/LRA.2022.3181374

### D.2 Sampling-based motion planning (RRT, RRT-Connect, PRM, RRT*)
- **Assumes:** The task reduces to collision-free transit between known start and goal configurations. Collision can be checked pointwise, and free space is "expansive" (random samples connect it with reasonable probability).
- **Executable form:** RRT-Connect (two trees with greedy connect) for single queries. PRM (a reusable roadmap) for many queries in a static scene. RRT* when path cost matters. Then shortcut-smooth the path. Collision checking uses primitive shapes (spheres, boxes, OBBs) built from the object-centric state.
- **Numpy-only feasible:** yes for low dimension (2–7 DoF). Write analytic distance functions and use `scipy.spatial.cKDTree` for nearest neighbours; vectorize edge checks by batch-interpolating configurations. The simulator cannot answer "is q in collision?" without `set_state`, so the agent keeps its own geometric model and executes only the final path in the simulator.
- **Describes the skill to a caller as:** "transit(goal): collision-free path found / not found within budget B; path length and minimum clearance. Assumes static obstacles and no intended contact."
- **Helps when / fails when:** Helps in clutter with multiple routes, where demos do not cover the new obstacle layout. / Fails when contact is intended (pushing counts as collision), in very narrow passages, when dynamics matter, or when the collision model is wrong.
- **How to test it:** In new scenes with added obstacles, compare the planner against a demo trajectory re-targeted to the new start and goal, and against straight-line interpolation. Metrics: collision rate, success and path length.
- **References:**
  - Kavraki, L. E. et al. (1996). Probabilistic roadmaps for path planning in high-dimensional configuration spaces. IEEE Trans. Robotics and Automation 12(4). doi:10.1109/70.508439
  - Kuffner, J. J. et al. (2000). RRT-Connect: An efficient approach to single-query path planning. ICRA 2000. doi:10.1109/ROBOT.2000.844730
  - Karaman, S. et al. (2011). Sampling-based algorithms for optimal motion planning. IJRR 30(7). arXiv:1105.1186, doi:10.1177/0278364911406761
  - Thomason, W. et al. (2024). Motions in microseconds via vectorized sampling-based planning. ICRA 2024. arXiv:2309.14545

### D.3 Trajectory optimization (CHOMP, TrajOpt, KOMO)
- **Assumes:** A good motion is a local optimum of smoothness plus constraint costs (goal, clearance, joint limits, orientation), and it can be reached from a sensible initial guess such as a straight line or a demo.
- **Executable form:** Discretize into T waypoints. Minimize Σ‖q_{t+1} − 2q_t + q_{t−1}‖² + w·Σ c_obs(q_t) subject to the endpoint and path constraints. CHOMP uses covariant gradient steps preconditioned by the smoothness metric. TrajOpt uses sequential convex programming with a trust region and hinge penalties on signed distance. KOMO exploits the k-order Markov (banded) structure in a Gauss-Newton / augmented-Lagrangian solver.
- **Numpy-only feasible:** yes. Use `scipy.optimize.minimize` (SLSQP, L-BFGS-B) over T×d variables, or a hand-written Gauss-Newton step with `scipy.linalg.solveh_banded`, plus analytic distance fields for primitive shapes. Warm-starting from a demo trajectory keeps the demo's "style".
- **Describes the skill to a caller as:** "smooth trajectory to goal satisfying constraints C; reports residual constraint violation, cost and convergence flag. Local method: may fail from a bad initial guess."
- **Helps when / fails when:** Helps for smooth motions with explicit constraints (keep upright, pass via a point, maintain clearance), especially when warm-started from demos. / Fails from local minima in clutter and on non-smooth contact switches or discontinuous costs.
- **How to test it:** Compare demo-initialized optimization, straight-line-initialized optimization and D.2 plus shortcutting in new scenes. Metrics: success, constraint violation, jerk.
- **References:**
  - Ratliff, N. et al. (2009). CHOMP: Gradient optimization techniques for efficient motion planning. ICRA 2009. doi:10.1109/ROBOT.2009.5152817
  - Schulman, J. et al. (2014). Motion planning with sequential convex optimization and convex collision checking. IJRR 33(9). doi:10.1177/0278364914528132
  - Toussaint, M. (2014). Newton methods for k-order Markov constrained motion problems (KOMO). arXiv:1407.0414
  - Sundaralingam, B. et al. (2023). cuRobo: Parallelized collision-free minimum-jerk robot motion generation. arXiv:2310.17274 (companion ICRA 2023 paper: doi:10.1109/ICRA48891.2023.10160765)

### D.4 Sampling-based MPC (MPPI, CEM / iCEM, predictive sampling)
- **Assumes:** A forward model (simulator, analytic or learned) predicts the outcomes of short action sequences well enough, and the task can be written as a cost. Replanning every step corrects model error.
- **Executable form:** Keep a nominal sequence U (H×m). Sample N perturbations (Gaussian, low-pass/colored noise, or spline knots), roll them out and score costs J_i.
  - **MPPI:** U ← U + Σ w_i ε_i with w_i ∝ exp(−(J_i − min J)/λ).
  - **CEM:** refit the mean and std to the top-k elites for a few iterations. iCEM adds colored noise, elite carry-over and shift-warm-start.
  - Execute the first action and shift U.
- **Numpy-only feasible:** partly.
  - (a) Standard per-step MPC needs a **numpy surrogate** vectorized over N samples: the D.20 push model, D.6 identified physics, or kNN/linear dynamics fit to demos and probes.
  - (b) The real simulator works only through reset plus prefix replay. That makes it practical for **whole-episode open-loop CEM** over a low-dimensional parameterization (D.17 primitive parameters, D.14 goals), not for per-step MPC.
- **Describes the skill to a caller as:** "closed-loop cost minimizer: goal = cost J; horizon H, N samples, model M. Reports predicted vs realized cost and one-step model error, which indicates when the skill is outside its model's validity."
- **Helps when / fails when:** Helps for underactuated or contact-rich tasks (pushing) with a cheap, adequate model and an easy-to-write cost. / Fails with sparse costs over long horizons, biased models, or high-dimensional actions on a small sample budget.
- **How to test it:** Compare replanning against executing the first plan open-loop, and the surrogate model against simulator-replay CEM. Metrics: success, final object pose error, wall-clock per step. Sweep N and H.
- **References:**
  - Williams, G. et al. (2017). Information theoretic MPC for model-based reinforcement learning. ICRA 2017. doi:10.1109/ICRA.2017.7989202
  - Pinneri, C. et al. (2020). Sample-efficient cross-entropy method for real-time planning. CoRL 2020 (PMLR 155). arXiv:2008.06389
  - Howell, T. et al. (2022). Predictive Sampling: Real-time behaviour synthesis with MuJoCo. arXiv:2212.00541
  - Hansen, N. et al. (2024). TD-MPC2: Scalable, robust world models for continuous control. ICLR 2024. arXiv:2310.16828

### D.5 Gradient-based optimal control (iLQR / DDP)
- **Assumes:** Dynamics are locally smooth and the cost is locally quadratic. A locally optimal nominal trajectory plus time-varying linear feedback captures the skill.
- **Executable form:**
  1. Roll out the nominal trajectory.
  2. Linearize the dynamics (A_t, B_t) and quadratize the cost.
  3. Run a backward Riccati pass to get the feedforward k_t and gains K_t.
  4. Do a forward pass with line search and Levenberg-Marquardt regularization. Handle box limits with a projected-Newton (control-limited) solve.
  5. Output the policy u_t = ū_t + K_t(x_t − x̄_t).
- **Numpy-only feasible:** partly. It needs a numpy model with Jacobians. Get them analytically, by vectorized central finite differences of the surrogate, or by local linear regression on simulator rollouts around the nominal trajectory (via replay). There is no autodiff.
- **Describes the skill to a caller as:** "tracks nominal x̄_t with gains K_t; valid while ‖x − x̄‖ < r (measured); reports expected cost-to-go."
- **Helps when / fails when:** Helps with smooth dynamics, moderate state dimension, and when feedback around a nominal is needed. / Fails across contact-mode switches (zero or misleading gradients), from poor initializations, and under hard state constraints.
- **How to test it:** Compare feedback (K_t) against open-loop ū_t under initial-state variation across new scenes, and against MPPI at equal compute. Metrics: success, cost, tracking error.
- **References:**
  - Li, W. et al. (2004). Iterative linear quadratic regulator design for nonlinear biological movement systems. ICINCO 2004. doi:10.5220/0001143902220229
  - Tassa, Y. et al. (2012). Synthesis and stabilization of complex behaviors through online trajectory optimization. IROS 2012. doi:10.1109/IROS.2012.6386025
  - Tassa, Y. et al. (2014). Control-limited differential dynamic programming. ICRA 2014. doi:10.1109/ICRA.2014.6907001
  - Mastalli, C. et al. (2020). Crocoddyl: An efficient and versatile framework for multi-contact optimal control. ICRA 2020. arXiv:1909.04947

### D.6 Real2sim, system identification and digital twins
- **Assumes:** The environment follows a parametric physics model. Identifying a few physical parameters (friction, mass, center of mass, damping, geometry) generalizes better than fitting a black-box policy or model on the same data.
- **Executable form:**
  1. Write a numpy model with parameters θ.
  2. Gather transitions from demos plus exploratory probes. Choose probes that excite the parameters (ASID maximizes Fisher information).
  3. Fit θ by `least_squares` on predicted vs observed next states, or get a posterior by likelihood-free inference (BayesSim).
  4. Plan or train over posterior samples: a "digital twin" or "digital cousins" (RialTo, ACDC).
- **Numpy-only feasible:** yes. The agent cannot change the simulator's parameters, but it can calibrate its own surrogate against the simulator. Get a posterior by bootstrap or grid search and use the samples as an ensemble in D.4.
- **Describes the skill to a caller as:** "model-based: quasi-static push model, μ = 0.32 ± 0.04, 1-step error e on fresh probes; valid for object sizes and speeds within the probed range."
- **Helps when / fails when:** Helps when demos are few but probes are cheap, and physical parameters vary across scenes. / Fails with unmodeled effects (toppling, rolling, compliance), unidentifiable parameters (poor excitation), or the wrong model class.
- **How to test it:** Collect new-scene probe episodes. Compare the multi-step prediction error and downstream MPC success of the identified θ, a nominal θ, and a black-box regressor fit on the same data.
- **References:**
  - Ramos, F. et al. (2019). BayesSim: Adaptive domain randomization via probabilistic inference for robotics simulators. RSS 2019. arXiv:1906.01728
  - Memmel, M. et al. (2024). ASID: Active exploration for system identification in robotic manipulation. ICLR 2024. arXiv:2404.12308
  - Torne Villasevil, M. et al. (2024). Reconciling reality through simulation: A real-to-sim-to-real approach for robust manipulation (RialTo). RSS 2024. arXiv:2403.03949
  - Dai, T. et al. (2024). Automated creation of digital cousins for robust policy learning (ACDC). CoRL 2024 (PMLR 270). arXiv:2410.07408

### D.7 Contact-implicit optimization and differentiable/smoothed physics
- **Assumes:** The contact schedule (which face, when, sticking or sliding) need not be fixed in advance. It can be found by optimizing over trajectories with contact forces as variables under complementarity constraints, or through a smoothed contact model whose gradients are informative.
- **Executable form:** Decision variables are (x_t, u_t, λ_t). Constraints: dynamics, gap φ(q) ≥ 0, λ ≥ 0, relaxed complementarity φ·λ ≤ ε, and friction cones; solve with SQP or augmented Lagrangian (Posa). Alternatives:
  - Contact-invariant soft costs (CIO).
  - Smoothed quasi-dynamic contact (barrier smoothing κ) or randomized smoothing (Pang).
  - When first-order gradients through stiff contact are biased or high-variance, use zeroth-order gradient estimates from perturbed rollouts (Suh).
- **Numpy-only feasible:** partly. There is no differentiable simulator. Write a small planar contact model in numpy and solve with `scipy.optimize` (SLSQP, trust-constr) for few contacts and short horizons. Randomized smoothing is simple on a vectorized surrogate.
- **Describes the skill to a caller as:** "planner discovers the contact sequence (e.g., push face B, then re-contact on face A); reports the mode sequence and residual complementarity violation."
- **Helps when / fails when:** Helps when the demo's contact choice is infeasible in a new scene and a different contact must be discovered. / Fails on large problems (slow, local minima) and with friction or geometry mismatch.
- **How to test it:** In new scenes where the demo's contact face is blocked, compare a fixed contact schedule copied from the demo against contact-implicit discovery. Metrics: success, number of mode switches, solve time. For the smoothed model, also compare first-order and zeroth-order gradient variance.
- **References:**
  - Mordatch, I. et al. (2012). Discovery of complex behaviors through contact-invariant optimization. ACM Trans. Graphics 31(4) (SIGGRAPH). doi:10.1145/2185520.2185539
  - Posa, M. et al. (2014). A direct method for trajectory optimization of rigid bodies through contact. IJRR 33(1). doi:10.1177/0278364913506757
  - Suh, H. J. T. et al. (2022). Do differentiable simulators give better policy gradients? ICML 2022 (PMLR 162). arXiv:2202.00817
  - Pang, T. et al. (2023). Global planning for contact-rich manipulation via local smoothing of quasi-dynamic contact models. IEEE T-RO. arXiv:2206.10787, doi:10.1109/TRO.2023.3300230

### D.8 Task and motion planning (PDDLStream, logic-geometric programming)
- **Assumes:** Long-horizon tasks factor into a discrete plan of symbolic operators (preconditions and effects) whose continuous parameters (grasps, placements, contact points, paths) are found by samplers or optimization. Geometric failure is fed back to the discrete search.
- **Executable form:**
  - **PDDLStream:** a PDDL domain plus "streams", which are conditional generators (sample-grasp, IK, plan-motion, collision test). Planning is optimistic, with incremental binding.
  - **LGP:** enumerate symbolic skeletons by tree search. Each skeleton defines one nonlinear program over the whole trajectory (KOMO) with mode constraints (stable, touch, push). Solve the cheapest skeletons first; stable-mode physics also covers tool use.
- **Numpy-only feasible:** partly. For small domains, write a Python forward search (BFS or A*) over a handful of operators, numpy samplers and checkers as streams, and `scipy.optimize` for each skeleton's program. No external PDDL planner is needed.
- **Describes the skill to a caller as:** "composite skill: plan [push(B, clear), pick(A), place(A, goal)]; each step has pre/postconditions; on failure reports the step and cause (no IK, collision, infeasible)."
- **Helps when / fails when:** Helps for multi-object tasks with ordering or obstruction dependencies, and for generalization to more objects. / Fails when symbols are wrong or incomplete, branching is large, or effects are stochastic (contact-rich outcomes break the assumption of deterministic operator effects).
- **How to test it:** Compare one flat controller (single MPC or regressed policy for the whole task) against TAMP decomposition, in new scenes with more objects or obstacles. Metrics: success vs object count, planning time.
- **References:**
  - Toussaint, M. (2015). Logic-geometric programming: An optimization-based approach to combined task and motion planning. IJCAI 2015. https://www.ijcai.org/Abstract/15/274
  - Toussaint, M. et al. (2018). Differentiable physics and stable modes for tool-use and manipulation planning. RSS 2018. doi:10.15607/RSS.2018.XIV.044
  - Garrett, C. R. et al. (2020). PDDLStream: Integrating symbolic planners and blackbox samplers via optimistic adaptive planning. ICAPS 2020. arXiv:1802.08705
  - Garrett, C. R. et al. (2021). Integrated task and motion planning. Annual Review of Control, Robotics, and Autonomous Systems 4. arXiv:2010.01083

### D.9 Learned symbolic operators, predicate invention and bilevel planning
- **Assumes:** Demos and transitions can be abstracted into a few relational predicates over objects such that each skill has compact lifted STRIPS-style effects. Planning in that abstraction and then refining continuous parameters generalizes compositionally.
- **Executable form:**
  1. Predicates are classifiers on object features, e.g., On(a,b) or InRegion(o,r) as thresholds on relative poses.
  2. Segment the demos into skill calls.
  3. Cluster transitions by lifted add/delete effects to get operators. Preconditions are the atoms true in all pre-states.
  4. Fit a sampler for each operator's continuous parameters from the demo parameters.
  5. Bilevel planning: A* over operators, then refine by sampling parameters and checking in the model or simulator, with backtracking.
  - **Predicate invention:** enumerate a feature grammar and select predicates by how much they improve planning (Silver 2023), or alternate symbolic effect learning with neural predicate fitting (IVNTR, Li 2025).
- **Numpy-only feasible:** yes. Predicates are numpy thresholds, operator learning is set arithmetic, samplers are Gaussian or kNN, and the planner is Python A*.
- **Describes the skill to a caller as:** "operator Push(o, r): pre {Reachable(o), PathClear(o, r)}, eff {+In(o, r)}, with predicate definitions and the sampler's refinement success rate."
- **Helps when / fails when:** Helps when test tasks have more objects or new goal conjunctions than the demos. / Fails when predicates miss the geometry that matters (refinement keeps failing), when demo segmentation is noisy, or on purely continuous precision tasks.
- **How to test it:** Build from demos with n objects and evaluate in scenes with n + k objects or new goals. Compare learned operators, hand-written predicates and no abstraction. Metrics: success, nodes expanded, refinement failures.
- **References:**
  - Konidaris, G. et al. (2018). From skills to symbols: Learning symbolic representations for abstract high-level planning. JAIR 61. doi:10.1613/jair.5575
  - Silver, T. et al. (2021). Learning symbolic operators for task and motion planning. IROS 2021. arXiv:2103.00589
  - Silver, T. et al. (2023). Predicate invention for bilevel planning. AAAI 2023. arXiv:2203.09634
  - Li, B. et al. (2025). Bilevel learning for bilevel planning. RSS 2025. arXiv:2502.08697

### D.10 LLM/VLM task planners with affordance grounding (SayCan, Inner Monologue)
- **Assumes:** Common-sense sequencing from a language model is useful only when it is grounded by per-skill feasibility (affordance/value) estimates and closed-loop feedback: success detection and scene state after each step.
- **Executable form:**
  - **SayCan:** choose the skill maximizing p_LM(skill | instruction, history) × V_skill(state).
  - **Inner Monologue:** after each step, feed back success/failure and the scene description, then replan.
  - **Text2Motion:** check the geometric feasibility of the whole sequence before executing it.
  - **LLM-Modulo:** the LLM proposes; external verifiers (model, simulator) accept or critique.
- **Numpy-only feasible:** partly. The coding agent is the LLM at authoring time; at runtime nothing calls an LLM. What remains executable is a skill library in which each skill exposes `affordance(state) → [0,1]` (e.g., logistic regression on reachability or clearance features, fit from new-scene attempts) and `succeeded(state)`, plus a small planner that ranks or filters skills by them.
- **Describes the skill to a caller as:** "affordance(state) = estimated success probability (with its calibration); success detector = predicate P; the caller ranks candidate skills by these."
- **Helps when / fails when:** Helps for long instructions over a large skill library, and for recovery via feedback. / Fails when affordances are miscalibrated, when plans violate geometric constraints the affordances do not capture, and on purely continuous tasks.
- **How to test it:** Use scenes where some skills are infeasible. Compare the planner with and without affordance weighting and with and without step feedback. Metrics: task success, number of infeasible skill attempts, Brier score of affordances.
- **References:**
  - Ahn, M. et al. (2022). Do As I Can, Not As I Say: Grounding language in robotic affordances (SayCan). CoRL 2022 (PMLR 205). arXiv:2204.01691
  - Huang, W. et al. (2022). Inner Monologue: Embodied reasoning through planning with language models. CoRL 2022 (PMLR 205). arXiv:2207.05608
  - Lin, K. et al. (2023). Text2Motion: From natural language instructions to feasible plans. Autonomous Robots 47. arXiv:2303.12153, doi:10.1007/s10514-023-10131-7
  - Kambhampati, S. et al. (2024). LLMs can't plan, but can help planning in LLM-Modulo frameworks. ICML 2024 (PMLR 235). arXiv:2402.01817

### D.11 Behavior trees
- **Assumes:** Task logic is a modular tree of conditions and actions, re-evaluated every tick. Each node returns SUCCESS, FAILURE or RUNNING, and composition uses Sequence, Fallback (Selector) and Parallel nodes.
- **Executable form:** Small Python node classes, ticked at the control rate. The canonical postcondition–precondition–action pattern is Fallback(GoalMet?, Sequence(Pre₁?, …, Action)). Backchaining replaces a failing precondition with a subtree that achieves it; Styrud 2025 automates the expansion with an LLM. Trees can also be learned by search or genetic programming (Colledanchise 2019).
- **Numpy-only feasible:** yes (pure Python).
- **Describes the skill to a caller as:** "tree Fallback(AtGoal?, Sequence(Reachable?, Approach, Push)); returns status each tick; reports the failing node; retries via fallbacks."
- **Helps when / fails when:** Helps with reactivity (objects moved, a sub-skill failed), retries, readable composition and failure attribution. / Fails when conditions are hard to define from state, fallbacks oscillate, or finding a better behavior needs optimization (a BT only sequences).
- **How to test it:** Inject failures by corrupting one sub-skill's parameters with probability p, or use scenes whose initial states force a precondition to be false. Compare the BT against the same sub-skills in a linear script. Metrics: recovery rate, success, extra steps.
- **References:**
  - Colledanchise, M. et al. (2018). Behavior Trees in Robotics and AI: An Introduction. CRC Press. arXiv:1709.00084, doi:10.1201/9780429489105
  - Colledanchise, M. et al. (2019). Learning of behavior trees for autonomous agents. IEEE Trans. Games. arXiv:1504.05811, doi:10.1109/TG.2018.2816806
  - Iovino, M. et al. (2022). A survey of behavior trees in robotics and AI. Robotics and Autonomous Systems 154. arXiv:2005.05842
  - Styrud, J. et al. (2025). Automatic behavior tree expansion with LLMs for robotic manipulation. ICRA 2025. arXiv:2409.13356

### D.12 Finite-state machines and hybrid automata (mode switching)
- **Assumes:** Behavior consists of discrete modes (approach, contact, push-sticking, push-sliding, retreat), each with its own continuous controller or dynamics. Transitions fire on guard conditions over the state (contact, thresholds on distance or velocity).
- **Executable form:** A dictionary mapping each mode to its controller, guard functions and reset maps. To learn one from demos:
  - Segment demos at change points in features (contact on/off, velocity regime); Niekum uses BP-AR-HMM.
  - Fit a controller per mode, and fit guards as classifiers on the states at transitions.
  - Hybrid MPC can plan over modes explicitly, e.g., sticking vs sliding pusher-slider modes (Hogan & Rodriguez).
  - Counterfactual perturbations of the demos show where the true mode boundaries lie (Wang 2024).
- **Numpy-only feasible:** yes. Change points come from dynamic programming over piecewise-linear fits. Guards are thresholds or logistic regression (scipy).
- **Describes the skill to a caller as:** "modes [approach → contact → push → retreat], with a guard on each transition (e.g., contact when gap < 5 mm); reports the current mode; raises an error mode on an unexpected guard."
- **Helps when / fails when:** Helps when demos have clear phases or contact changes, and gives interpretable failures (stuck in a mode). / Fails with blended or gradual phases, wrong guards (chattering), or many modes.
- **How to test it:** Compare one global controller fit to whole demos against the mode-segmented controller, in new scenes. Metrics: success, and precision/recall of predicted transition times against simulator contact events.
- **References:**
  - Henzinger, T. A. (1996). The theory of hybrid automata. LICS 1996. doi:10.1109/LICS.1996.561342
  - Niekum, S. et al. (2015). Learning grounded finite-state representations from unstructured demonstrations. IJRR 34(2). doi:10.1177/0278364914554471
  - Hogan, F. R. et al. (2020). Reactive planar non-prehensile manipulation with hybrid model predictive control. IJRR 39(7). doi:10.1177/0278364920913938
  - Wang, Y. et al. (2024). Grounding language plans in demonstrations through counterfactual perturbations. ICLR 2024. arXiv:2403.17124

### D.13 Funnels and sequential composition (preconditions as regions of attraction)
- **Assumes:** Each skill is a feedback controller with a domain (region of attraction / preimage) and a goal set. Tasks are solved by chaining skills whose goal set lies inside the next skill's domain.
- **Executable form:**
  - For each skill, estimate its domain empirically: run it from many initial states over reset seeds, label success, and fit an inner approximation (ellipsoid or classifier).
  - At runtime, choose the skill whose domain contains the current state and that lies closest to the goal in the chaining graph.
  - LQR-trees grow trajectories with LQR funnels until the domain is covered. Funnel libraries precompute a set of funnels to compose online.
- **Numpy-only feasible:** yes for empirical regions: Monte Carlo over resets, then a covariance ellipsoid or logistic-regression boundary. SOS verification needs an SDP solver, which scipy lacks, so no.
- **Describes the skill to a caller as:** "precondition: {x : g(x) ≤ 0} with measured success ≥ 90% inside; postcondition: terminal set T; composable after any skill whose T ⊂ my precondition."
- **Helps when / fails when:** Helps with chaining, deciding when a skill applies, and honest capability reporting. / Fails when the state dimension makes region estimation sample-hungry, or when conditions drift between scenes.
- **How to test it:** Compare chaining with and without pre/post containment checks (random or fixed order vs funnel-based selection). Metrics: chain success, and precision/recall of the predicted domain on new-scene starts.
- **References:**
  - Burridge, R. R. et al. (1999). Sequential composition of dynamically dexterous robot behaviors. IJRR 18(6). doi:10.1177/02783649922066385
  - Tedrake, R. et al. (2010). LQR-trees: Feedback motion planning via sums-of-squares verification. IJRR 29(8). doi:10.1177/0278364910369189
  - Majumdar, A. et al. (2017). Funnel libraries for real-time robust feedback motion planning. IJRR 36(8). arXiv:1601.04037, doi:10.1177/0278364917712421
  - Chen, Y. et al. (2023). Sequential Dexterity: Chaining dexterous policies for long-horizon manipulation. CoRL 2023 (PMLR 229). arXiv:2309.00987

### D.14 Dynamic Movement Primitives (DMPs)
- **Assumes:** A motion is a stable second-order attractor to a goal g, plus a learned forcing term that depends on a phase variable and decays to zero. New start or goal positions rescale the demo shape, and τ rescales time. The motion converges to g regardless of the forcing term.
- **Executable form:** τ²ÿ = α(β(g − y) − τẏ) + f(s)(g − y₀), with τṡ = −α_s s and f(s) = Σψ_i(s) w_i s / Σψ_i(s). Fit w by locally weighted regression on one demo and integrate with Euler steps. Coupling terms add obstacle avoidance, force feedback or phase stopping (slow the phase when tracking error grows).
- **Numpy-only feasible:** yes (~40 lines).
- **Describes the skill to a caller as:** "DMP(g, τ): reproduces the demo shape toward goal g in duration τ; guaranteed convergence to g; K basis weights; phase-stopping on error."
- **Helps when / fails when:** Helps for reproducing a point-to-point motion from a single demo with shifted goals. / Fails when the new goal should change the shape rather than rescale it (g − y₀ ≈ 0 in the demo blows up the scaling), when the right quantity is force rather than position, or when several demos need a distribution.
- **How to test it:** In new scenes with new goals, compare the DMP against a translated demo replay and linear interpolation. Metrics: final error, success.
- **References:**
  - Pastor, P. et al. (2009). Learning and generalization of motor skills by learning from demonstration. ICRA 2009. doi:10.1109/ROBOT.2009.5152385
  - Ijspeert, A. J. et al. (2013). Dynamical movement primitives: Learning attractor models for motor behaviors. Neural Computation 25(2). doi:10.1162/NECO_a_00393
  - Saveriano, M. et al. (2023). Dynamic movement primitives in robotics: A tutorial survey. IJRR 42(13). arXiv:2102.03861, doi:10.1177/02783649231201196

### D.15 Probabilistic and via-point movement primitives (ProMP, VMP, ProDMP)
- **Assumes:** Demos are samples from a Gaussian distribution over basis-function weights. Low variance across demos marks what matters (e.g., the contact phase), and conditioning on via-points or goals yields new trajectories consistent with the demos.
- **Executable form:** y_t = Φ_tᵀw. Fit w per demo by ridge regression, then take the mean μ_w and covariance Σ_w across demos. Condition on a via-point (y*, t*) by Gaussian conditioning. Temporal modulation comes from phase. Primitives compose by co-activation or blending. VMP adds a via-point modulation term for extrapolation; ProDMP merges ProMP with DMP-style smooth initial conditions.
- **Numpy-only feasible:** yes.
- **Describes the skill to a caller as:** "trajectory distribution: mean and per-time variance (tight near contact, loose in transit); accepts via-points (y*, t*, σ²) and goals within the demo spread."
- **Helps when / fails when:** Helps with several demos that vary meaningfully, and when conditioning on new via-points or goals. / Fails with one demo (no covariance), multimodal demos (the Gaussian averages the modes), or far extrapolation.
- **How to test it:** In new scenes, compare conditioning on the new goal against a DMP with a changed goal and the nearest demo. Metrics: success, final error, and the fraction of the trajectory outside ±2σ.
- **References:**
  - Paraschos, A. et al. (2013). Probabilistic movement primitives. NeurIPS 2013. https://papers.nips.cc/paper_files/paper/2013/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html
  - Paraschos, A. et al. (2018). Using probabilistic movement primitives in robotics. Autonomous Robots 42. doi:10.1007/s10514-017-9648-7
  - Zhou, Y. et al. (2019). Learning via-point movement primitives with inter- and extrapolation capabilities. IROS 2019. doi:10.1109/IROS40897.2019.8968586
  - Li, G. et al. (2023). ProDMP: A unified perspective on dynamic and probabilistic movement primitives. IEEE RA-L. arXiv:2210.01531, doi:10.1109/LRA.2023.3248443

### D.16 Waypoint / keyframe primitives
- **Assumes:** A demo is captured by a few key poses (at gripper open/close events, velocity near zero, or contact changes), with a simple interpolator or planner between them. The policy decides the next waypoint instead of dense actions, which reduces compounding error.
- **Executable form:**
  - Keyframe heuristics (gripper change, zero velocity), as in Q-attention and PerAct.
  - AWE: dynamic programming picks the fewest waypoints whose linear interpolation reconstructs the demo within ε.
  - Express each waypoint in the frame of its relevant object, then execute it with a controller (D.18/D.19) or a planner (D.2/D.3).
- **Numpy-only feasible:** yes.
- **Describes the skill to a caller as:** "k waypoints relative to object O (tolerance ε); interpolation between them is linear or planned; reports reconstruction error and which waypoint failed."
- **Helps when / fails when:** Helps with long demos and generalization to new object poses (via object-relative waypoints). / Fails when the path between waypoints matters (sliding pushes, dynamic motions) or timing is critical.
- **How to test it:** Compare dense per-step regression from demos against the waypoint policy, in new scenes. Sweep ε. Metrics: success, decisions per episode.
- **References:**
  - Akgun, B. et al. (2012). Keyframe-based learning from demonstration. International Journal of Social Robotics 4. doi:10.1007/s12369-012-0160-0
  - James, S. et al. (2022). Q-attention: Enabling efficient learning for vision-based robotic manipulation. IEEE RA-L 7(2). arXiv:2105.14829
  - Shridhar, M. et al. (2022). Perceiver-Actor: A multi-task transformer for robotic manipulation. CoRL 2022 (PMLR 205). arXiv:2209.05451
  - Shi, L. X. et al. (2023). Waypoint-based imitation learning for robotic manipulation (AWE). CoRL 2023 (PMLR 229). arXiv:2307.14326

### D.17 Parameterized action primitives (RAPS, MAPLE, PRIME)
- **Assumes:** The action space is better expressed as a small library of primitives (reach, grasp, push, lift, gripper open/close), each with a few continuous parameters. The policy chooses a (primitive, parameters) pair, as in a parameterized-action MDP.
- **Executable form:** Each primitive is a closed-loop `env.step` routine with arguments, e.g., push(contact_pt, direction, distance), and a termination test. The top level selects a primitive and its parameters by CEM or search in the simulator, by classifiers fit to demos, or by RL. PRIME parses demos into primitive sequences and imitates at the primitive level. MAPLE keeps an "atomic" low-level primitive to cover expressivity gaps.
- **Numpy-only feasible:** yes. Primitives are Python loops. Because each episode has few decisions, whole-episode reset-and-replay CEM over primitive parameters is affordable in the real simulator.
- **Describes the skill to a caller as:** "push(p, d, L): go to pre-contact 2 cm behind p, push L along d, retract; parameter ranges; expected object displacement; termination condition."
- **Helps when / fails when:** Helps with sparse rewards, long horizons, sample efficiency and interpretability. / Fails when the library lacks a needed behavior or the parameterization is poor (the expressivity gap).
- **How to test it:** At an equal simulator-step budget, compare search or RL over primitives against search over raw actions. Metrics: success, steps to solve. Ablate one primitive at a time.
- **References:**
  - Masson, W. et al. (2016). Reinforcement learning with parameterized actions. AAAI 2016. arXiv:1509.01644
  - Dalal, M. et al. (2021). Accelerating robotic reinforcement learning via parameterized action primitives (RAPS). NeurIPS 2021. arXiv:2110.15360
  - Nasiriany, S. et al. (2022). Augmenting reinforcement learning with behavior primitives for diverse manipulation tasks (MAPLE). ICRA 2022. arXiv:2110.03655
  - Gao, T. et al. (2024). PRIME: Scaffolding manipulation tasks with behavior primitives for data-efficient imitation learning. IEEE RA-L. arXiv:2403.00929, doi:10.1109/LRA.2024.3443610

### D.18 Impedance, admittance and operational-space control
- **Assumes:** In contact, the controller should regulate the dynamic relation between motion and force, i.e., a virtual spring-damper F = K(x_d − x) + D(ẋ_d − ẋ), rather than position alone. Operational-space control (OSC) decouples task-space dynamics using the task-space inertia Λ(q).
- **Executable form:**
  - **Impedance:** τ = Jᵀ(K e + D ė) + g(q).
  - **Admittance** (for position-controlled robots): integrate M ẍ = F_ext − Dẋ − K(x − x_d) into position commands.
  - **OSC:** τ = JᵀΛ(ẍ_d + K_p e + K_d ė) + null-space terms.
  - **Learned variants:** the policy outputs (Δx, K) (VICES) or predicts per-step stiffness (ACP).
- **Numpy-only feasible:** partly. It depends on the action space. With torque control plus a dynamics model, OSC is implementable. With end-effector position deltas, emulate compliance in the command layer: limit the commanded offset to e_max = F_max/K, and use contact force if it is in the state or infer contact from object motion.
- **Describes the skill to a caller as:** "compliant move: stiffness K per axis; peak force ≈ K·e_max; tolerates ±e position error in contact; stiff in free space, soft along the contact normal."
- **Helps when / fails when:** Helps in contact-rich tasks with position uncertainty (pressing against walls or fixtures, sliding along surfaces). / Fails when precise free-space tracking is needed (low K leaves tracking error), or when neither force signals nor a dynamics model is available.
- **How to test it:** In new scenes with goal-position noise, compare pure position commands, fixed stiffness and variable stiffness. Metrics: success, peak contact force or object penetration, sensitivity to the offset.
- **References:**
  - Hogan, N. (1985). Impedance control: An approach to manipulation: Part I—Theory. J. Dynamic Systems, Measurement, and Control 107(1). doi:10.1115/1.3140702
  - Khatib, O. (1987). A unified approach for motion and force control of robot manipulators: The operational space formulation. IEEE J. Robotics and Automation 3(1). doi:10.1109/JRA.1987.1087068
  - Martín-Martín, R. et al. (2019). Variable impedance control in end-effector space: An action space for reinforcement learning in contact-rich tasks (VICES). IROS 2019. arXiv:1906.08880
  - Hou, Y. et al. (2025). Adaptive compliance policy: Learning approximate compliance for diffusion guided control. ICRA 2025. arXiv:2410.09309

### D.19 Hybrid force/position control, task-frame formalism and guarded moves
- **Assumes:** In a task frame, each axis is either position-controlled or force-controlled according to natural constraints (you cannot move into a surface, and you cannot push along a free direction). Motions end on sensed events rather than at fixed poses, and such guarded moves absorb geometric uncertainty.
- **Executable form:**
  - **TFF spec:** for each axis, a velocity or force setpoint plus a stop condition, e.g., "move −z at 2 cm/s until F_z > 5 N or z < z_min."
  - **Hybrid control:** u = S·u_pos + (I − S)·u_force, with S the selection matrix.
  - **Guarded move:** `while not event(state) and t < T: step`.
  - **Fine-motion strategies:** chain guarded and compliant moves whose preimages cover the uncertainty.
- **Numpy-only feasible:** yes for guarded-move logic. Detect events from state predicates: object velocity onset, end-effector–object gap, force or contact flags if exposed. Hybrid force control requires force or torque access.
- **Describes the skill to a caller as:** "guarded move: advances along d until contact; reports contact position and whether it stopped on contact or on timeout/limit."
- **Helps when / fails when:** Helps with uncertain object size or pose, finding surfaces, and robustness to position noise in demos. / Fails without a reliable event signal, with fast motions that overshoot the stop, or with wrong constraint directions.
- **How to test it:** In new scenes with varied object sizes and heights, compare event-terminated moves against fixed-distance moves copied from demos. Metrics: success, overshoot or penetration, time.
- **References:**
  - Raibert, M. H. et al. (1981). Hybrid position/force control of manipulators. J. Dynamic Systems, Measurement, and Control 103(2). doi:10.1115/1.3139652
  - Lozano-Pérez, T. et al. (1984). Automatic synthesis of fine-motion strategies for robots. IJRR 3(1). doi:10.1177/027836498400300101
  - Bruyninckx, H. et al. (1996). Specification of force-controlled actions in the "task frame formalism"—a synthesis. IEEE Trans. Robotics and Automation 12(4). doi:10.1109/70.508440
  - Liu, W. et al. (2025). ForceMimic: Force-centric imitation learning with force-motion capture system for contact-rich manipulation. ICRA 2025. arXiv:2410.07554

### D.20 Quasi-static pushing mechanics (limit surface, motion cones)
- **Assumes:** Planar pushing is quasi-static: inertia is negligible, and the support friction's limit surface (often approximated as an ellipsoid) maps the applied wrench to object twist. The pusher contact sticks while the pusher velocity is inside the motion cone and slides otherwise. Together these give a closed-form object velocity for a given pusher motion.
- **Executable form:**
  - **Ellipsoid limit surface:** f_max = μ_s m g and m_max ≈ c·f_max·r (r a characteristic radius of the support area). The twist satisfies V ∝ H w with H = diag(1/f_max², 1/f_max², 1/m_max²).
  - **Per contact** (point p, normal n, friction μ_c): compute the motion-cone edges and classify the pusher velocity as sticking or sliding, which gives the object twist (Lynch 1992).
  - Use the model inside MPPI/CEM (D.4), as a mode-aware hybrid MPC (D.12), or for "stable pushing" choices that keep sticking contact. Fit μ_s, μ_c and c from probes (D.6).
- **Numpy-only feasible:** yes. The model is a few lines of linear algebra and vectorizes over samples.
- **Describes the skill to a caller as:** "push(face, dir): predicted object twist; stable (sticking) while dir is inside the motion cone (half-angle θ); reports predicted vs actual displacement and the identified friction."
- **Helps when / fails when:** Helps for slow tabletop pushing with few demos and new object shapes or poses. / Fails for fast (dynamic) pushes, unknown or non-uniform pressure distributions, toppling, or multi-object contact.
- **How to test it:** On pushes from new scenes, compare one-step and multi-step prediction error of the analytic model against a kNN or linear regressor on the same data. Then compare MPC success with each model. Ablate identified vs default friction.
- **References:**
  - Mason, M. T. (1986). Mechanics and planning of manipulator pushing operations. IJRR 5(3). doi:10.1177/027836498600500303
  - Lynch, K. M. et al. (1992). Manipulation and active sensing by pushing using tactile feedback. IROS 1992. doi:10.1109/IROS.1992.587370
  - Chavan-Dafle, N. et al. (2018). In-hand manipulation via motion cones. RSS 2018. arXiv:1810.00219, doi:10.15607/RSS.2018.XIV.058
  - Graesdal, B. P. et al. (2024). Towards tight convex relaxations for contact-rich manipulation. RSS 2024. arXiv:2402.10312

### D.21 Grasp analysis (friction cones, form/force closure, quality metrics)
- **Assumes:** Grasp stability follows from contact geometry and Coulomb friction. A grasp is force closure if the convex hull of the contact wrenches (friction cones linearized into k edges) contains the origin in its interior. Quality is the radius of the largest wrench ball inside the hull (Ferrari–Canny ε).
- **Executable form:**
  - **General grasps:** from contact points and normals on the object geometry, build the wrench matrix W (6 × nk), take its convex hull, and let ε be the minimum distance from the origin to a facet.
  - **Parallel-jaw grasps:** use the antipodal test: the angle between the contact line and each normal is below arctan μ.
  - **Robust variant:** average quality under pose and friction noise (Dex-Net).
  - **Optimization:** a differentiable force-closure energy (DexGraspNet).
- **Numpy-only feasible:** yes, with `scipy.spatial.ConvexHull` and object geometry from state primitives.
- **Describes the skill to a caller as:** "grasp(p1, p2): antipodal within friction half-angle arctan μ; ε = 0.12; succeeds under ±5 mm offset in 90% of perturbations."
- **Helps when / fails when:** Helps when choosing grasps on new shapes or poses and rejecting bad ones before execution. / Fails when geometry or friction is unknown, and when the simulator's gripper is not friction-based (suction, attach-on-contact), which makes closure metrics irrelevant.
- **How to test it:** In new scenes, compare top-ε grasps, random grasps and demo grasps transformed into the object frame. Metrics: grasp success, rank correlation of ε with success.
- **References:**
  - Nguyen, V.-D. (1988). Constructing force-closure grasps. IJRR 7(3). doi:10.1177/027836498800700301
  - Ferrari, C. et al. (1992). Planning optimal grasps. ICRA 1992. doi:10.1109/ROBOT.1992.219918
  - Mahler, J. et al. (2017). Dex-Net 2.0: Deep learning to plan robust grasps with synthetic point clouds and analytic grasp metrics. RSS 2017. arXiv:1703.09312
  - Wang, R. et al. (2023). DexGraspNet: A large-scale robotic dexterous grasp dataset for general objects based on simulation. ICRA 2023. arXiv:2210.02697

### D.22 Keypoint / constraint-based policies (kPAM, KETO, VoxPoser, ReKep; tool use)
- **Assumes:** Task success can be stated as geometric costs or constraints on a few semantic keypoints on objects, tools and the gripper, e.g., "tool tip within 1 cm of the target, handle vertical." Actions come from solving for end-effector or object poses that satisfy these constraints, re-solved in closed loop, and then executed by a low-level controller. Foundation models author the constraints or value maps as code.
- **Executable form:**
  - Define keypoints in object frames and write per-stage constraint functions c_i(keypoints) ≤ 0: subgoal constraints plus path constraints (ReKep).
  - Solve min cost s.t. c_i ≤ 0 with SLSQP at each step.
  - Tool use: grasp, function and effect keypoints (KETO).
  - VoxPoser instead composes 3D cost or affordance maps and plans over them.
- **Numpy-only feasible:** yes. Keypoints are exact from object poses. The coding agent takes the VLM's role, deriving constraints from the invariants of the demos' final and stage states (e.g., |p_obj − p_goal| < δ holds at the end of every demo).
- **Describes the skill to a caller as:** "stages with constraints: S1 {gripper at handle}, S2 {tool tip on target, tool upright}; reports each constraint's residual; success means all residuals ≤ 0."
- **Helps when / fails when:** Helps with category-level generalization (new poses and sizes), tool use, and explicit success specs. / Fails when success depends on dynamics that constraints omit (sliding friction in pushing), when keypoints are wrong, or on non-convex local minima.
- **How to test it:** In new scenes with new object poses and sizes, compare constraint solving against demo end-effector poses replayed in world frame and in object frame. Metrics: success, residuals.
- **References:**
  - Manuelli, L. et al. (2019). kPAM: KeyPoint affordances for category-level robotic manipulation. ISRR 2019 (Springer Proc. Adv. Robotics, 2022). arXiv:1903.06684
  - Qin, Z. et al. (2020). KETO: Learning keypoint representations for tool manipulation. ICRA 2020. arXiv:1910.11977
  - Huang, W. et al. (2023). VoxPoser: Composable 3D value maps for robotic manipulation with language models. CoRL 2023 (PMLR 229). arXiv:2307.05973
  - Huang, W. et al. (2024). ReKep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation. CoRL 2024 (PMLR 270). arXiv:2409.01652

### D.23 Code-as-policies and verified skill libraries (CaP, ProgPrompt, Voyager, Demo2Code)
- **Assumes:** A policy is best expressed as a readable program that composes state queries and control primitives with ordinary control flow and arithmetic. Skills accumulate as named, verified, reusable functions (a library). Demos can be summarized into a specification and then compiled into code.
- **Executable form:**
  - Python over helper APIs such as `obj_pose`, `move_to`, `push`, `in_contact`, with explicit parameters and thresholds.
  - Refine iteratively on execution feedback: exceptions, success checks, self-verification.
  - The library stores signature, docstring (pre/postconditions), and test record; retrieval is by description.
  - ProgPrompt gives the LLM program-like prompts: available actions, objects and example programs with assertions.
- **Numpy-only feasible:** yes. This is the agents' native medium.
- **Describes the skill to a caller as:** "push_to(obj, goal, tol=0.01) -> bool; preconditions; succeeded 18/20 in new scenes; known failure: goal behind an obstacle."
- **Helps when / fails when:** Helps with compositional tasks, spatial arithmetic, reuse and transparent failure. / Fails with hand-tuned thresholds that do not transfer, untested branches, and hallucinated or mismatched APIs.
- **How to test it:** On a new composite task, compare reusing the library against generating from scratch, and remove the feedback/self-verification loop as an ablation. Metrics: success, iterations and simulator steps to reach it.
- **References:**
  - Liang, J. et al. (2023). Code as Policies: Language model programs for embodied control. ICRA 2023. arXiv:2209.07753, doi:10.1109/ICRA48891.2023.10160591
  - Singh, I. et al. (2023). ProgPrompt: Generating situated robot task plans using large language models. ICRA 2023. arXiv:2209.11302
  - Wang, H. et al. (2023). Demo2Code: From summarizing demonstrations to synthesizing code via extended chain-of-thought. NeurIPS 2023. arXiv:2305.16744
  - Wang, G. et al. (2024). Voyager: An open-ended embodied agent with large language models. TMLR 2024. arXiv:2305.16291

### D.24 Programmatic and neuro-symbolic policies (program synthesis, NS-CL, neuro-symbolic predicates)
- **Assumes:** A policy is a program in a small DSL. Its structure (loops, conditionals, symbolic operators) is discrete and interpretable, while its leaves (concept classifiers, thresholds, gains) are fit to data. Programs extrapolate to longer horizons, more objects and larger distances better than flat regressors. Neuro-symbolic predicates (NSPs) are code snippets that compute predicates, possibly calling learned components.
- **Executable form:**
  - Define a DSL (sensors, primitives, if/while).
  - Search programs by enumeration, LLM proposal, or latent program-space search (LEAPS), scored by simulator return.
  - PIRL: imitate an oracle, then tune constants by Bayesian or local optimization.
  - NS-CL: learned concept modules executed by a symbolic program.
  - VisualPredicator: invent NSPs online and learn abstract operators over them.
- **Numpy-only feasible:** yes for small DSLs. Learned leaves become logistic regression, thresholds or linear gains. The VLM calls inside NSPs become state predicates.
- **Describes the skill to a caller as:** "program: while not At(o, g): if not Contact(o): approach(o) else push(o, g − o); constants (k, δ) fitted; predicates defined as …"
- **Helps when / fails when:** Helps for out-of-distribution scene sizes and horizons, with few demos, when verifiability matters. / Fails when program search explodes, the DSL lacks a needed primitive, or behavior needs fine continuous control.
- **How to test it:** Build a program policy and a regression policy from the same demos. Evaluate both on out-of-distribution new scenes (more objects, larger distances). Metrics: success, program size.
- **References:**
  - Verma, A. et al. (2018). Programmatically interpretable reinforcement learning. ICML 2018 (PMLR 80). arXiv:1804.02477
  - Mao, J. et al. (2019). The Neuro-Symbolic Concept Learner: Interpreting scenes, words, and sentences from natural supervision. ICLR 2019. arXiv:1904.12584
  - Trivedi, D. et al. (2021). Learning to synthesize programs as interpretable and generalizable policies (LEAPS). NeurIPS 2021. arXiv:2108.13643
  - Liang, Y. et al. (2025). VisualPredicator: Learning abstract world models with neuro-symbolic predicates for robot planning. ICLR 2025. arXiv:2410.23156

---

## E. Task-structure and demonstration-structure priors

These priors get their structure from the task or from demonstrations: where demos split, what stays invariant, what counts as the goal, what must never happen, and how sure a skill is of itself. Throughout, "state" means the fully observed object-centric state (poses, velocities, gripper, contacts if exposed). "Sim" means the simulator, reachable through reset/step. A numpy-only version assumes Python + numpy + scipy, with no images and no torch.

---

### E.1 Waypoint / keyframe extraction
- **Assumes:** A demo is fully described by a short sequence of keyframes (waypoints) joined by a simple interpolator or controller. The information sits in the keyframes, not in the samples between them. Gripper toggles and velocity minima are the natural keyframes.
- **Executable form:** AWE: find the smallest subset of demo states such that linearly interpolating between them reconstructs every demo state (EE pose + gripper, optionally object-relative, see E.4) within error ε. Gripper toggles and the final state are always kept. PerAct / C2F-ARM heuristic: a step is a keyframe if the gripper state changes or the arm velocity is ≈ 0. The policy predicts or replays the next waypoint, and a P-controller/IK tracks the segment to it.
- **Numpy-only feasible:** yes. AWE is recursive splitting or O(T²) dynamic programming over point-to-segment distances (`np.linalg.norm`). The heuristic is `np.where((speed<δ) | (np.diff(grip)!=0))`. Execute through sim step with a P-controller toward each waypoint.
- **Describes the skill to a caller as:** "k=6 waypoints; waypoint i = (EE pose in frame of object A, gripper state); max reconstruction error ≤ 5 mm; now tracking waypoint 4 of 6."
- **Helps when / fails when:**
  - Helps: quasi-static pick/place/insert, where a shorter decision horizon reduces compounding error.
  - Fails: timing or force matters between waypoints (pouring, wiping, throwing), or a coarse ε drops a critical pre-grasp or pre-insert pose.
- **How to test it:** Use the same learner or replayer with dense per-step targets, then with AWE waypoints at ε ∈ {2, 5, 10, 20} mm. Metrics: success rate over new-seed sim scenes and decision-horizon length. Plot success against ε to find the knee.
- **References:**
  - Akgun, Cakmak, Jiang, Thomaz, "Keyframe-based Learning from Demonstration", Int. J. Social Robotics 2012. DOI 10.1007/s12369-012-0160-0
  - Shridhar, Manuelli, Fox, "Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation", CoRL 2022. arXiv:2209.05451 (keyframe heuristic)
  - Shi, Sharma, Zhao, Finn, "Waypoint-Based Imitation Learning for Robotic Manipulation" (AWE), CoRL 2023. arXiv:2307.14326
  - Mehta, Habibian, Losey, "Waypoint-Based Reinforcement Learning for Robot Manipulation Tasks", IROS 2024. arXiv:2403.13281

### E.2 Changepoint segmentation into phases and skills
- **Assumes:** Demos are concatenations of a few recurring sub-behaviours. Each has simple dynamics (roughly constant goal, or locally linear). Boundaries fall where the moving object, the contacts, the gripper state or the governing goal changes.
- **Executable form:** (i) Event features: gripper toggles, contact on/off, an object starting or stopping, a relative distance crossing a threshold. (ii) Statistical changepoints: fit piecewise-linear or AR segments and choose breaks that minimise residual + penalty per break (PELT / binary segmentation). (iii) Cluster the candidate transition states across demos (TSC) and keep only boundaries that recur. (iv) Align segments across demos by order or DTW, then label each segment with its target object and its exit event. Learned variants: CompILE (latent boundaries); UVD (a new subgoal starts where distance-to-goal in the embedding stops decreasing monotonically).
- **Numpy-only feasible:** yes. Event features are direct comparisons on the state. Changepoints come from DP over `np.linalg.lstsq` segment residuals. Clustering uses `scipy.cluster.vq.kmeans2`. UVD-style: run a monotonicity test on distance-to-final-state per feature subset.
- **Describes the skill to a caller as:** "phase 3 of 5 ('transport'); entered when A is grasped (gripper closed ∧ A.z > table + 2 cm); exits when A is within 2 cm above B."
- **Helps when / fails when:**
  - Helps: multi-stage tasks. Per-phase policies are simpler and reusable, and phase boundaries are natural checkpoints for verification (E.13, E.19).
  - Fails: demos use inconsistent orderings, phases overlap (two motions at once), or thresholds tuned on few demos over- or under-segment.
- **How to test it:** Train one monolithic policy and a set of per-phase policies with the same learner. Metrics: task success and per-phase success in new sim scenes, and segmentation consistency (fraction of demos with the same number and order of segments).
- **References:**
  - Niekum, Osentoski, Konidaris, Barto, "Learning and Generalization of Complex Tasks from Unstructured Demonstrations", IROS 2012. DOI 10.1109/IROS.2012.6386006
  - Krishnan et al., "Transition State Clustering: Unsupervised Surgical Trajectory Segmentation for Robot Learning", IJRR 2017. DOI 10.1177/0278364917743319
  - Kipf et al., "CompILE: Compositional Imitation Learning and Execution", ICML 2019. arXiv:1812.01483
  - Zhang et al., "Universal Visual Decomposer: Long-Horizon Manipulation Made Easy", ICRA 2024. arXiv:2310.08581

### E.3 Bottleneck states and option discovery
- **Assumes:** Good subgoals are states that many successful trajectories pass through (bottlenecks, high betweenness) or that separate weakly connected regions of the state space. Options that reach them are reusable across tasks.
- **Executable form:** Discretise or cluster the states from demos and sim rollouts, then build the transition graph. Score nodes by one of: (a) diverse density, meaning frequent in successful trajectories and rare in failed ones; (b) betweenness centrality; (c) extrema of the low-frequency eigenvectors of the graph Laplacian (eigenoptions); (d) spikes in state-action novelty, which mark decision points (NBDI). Build one option per subgoal: termination = within ε of the subgoal; initiation = the states from which sim rollouts reach it.
- **Numpy-only feasible:** yes, at moderate resolution. Use `scipy.sparse` adjacency, `scipy.sparse.csgraph` shortest paths or betweenness via BFS counts, and `np.linalg.eigh` of the Laplacian. Diverse density is frequency counting. For continuous states, k-means centroids serve as nodes.
- **Describes the skill to a caller as:** "reaches subgoal 'A held above B'. This bottleneck is traversed by 50/50 demos and 3% of random rollouts. Initiation = clusters {4, 7, 9}; typical duration 40 steps."
- **Helps when / fails when:**
  - Helps: long horizons with narrow passages (pre-grasp, doorway, pre-insert pose), and reuse across related tasks.
  - Fails: high-dimensional continuous state with no real bottleneck, discretisation aliasing, or "bottlenecks" that are artefacts of the exploration policy rather than of the task.
- **How to test it:** Compare hierarchical control over the discovered subgoals with flat control and with random subgoals. Metrics: success on new task variants, and sim steps needed to solve them.
- **References:**
  - McGovern, Barto, "Automatic Discovery of Subgoals in Reinforcement Learning using Diverse Density", ICML 2001.
  - Şimşek, Barto, "Skill Characterization Based on Betweenness", NeurIPS (NIPS) 2008. https://papers.nips.cc/paper/3411-skill-characterization-based-on-betweenness
  - Machado, Bellemare, Bowling, "A Laplacian Framework for Option Discovery in Reinforcement Learning", ICML 2017. arXiv:1703.00956
  - Kim et al., "NBDI: A Simple and Effective Termination Condition for Skill Extraction from Task-Agnostic Demonstrations", ICML 2025. arXiv:2501.12668

### E.4 Object-relative reference frames and demo transformation
- **Assumes:** Within each phase the motion is fixed relative to one object's frame (the grasped object or the target). The skill is therefore invariant to that object's pose. Cross-demo variance shows which frame governs each segment.
- **Executable form:** For each phase, express the EE (and held-object) trajectory in every candidate frame (world, each object) and pick the frame with the lowest cross-demo variance (the TP-GMM principle). At test time, re-anchor the segment to the new pose, `traj_new = T_obj_new · T_obj_demo⁻¹ · traj_demo`, and stitch segments with interpolated free-space transits (MimicGen / SkillMimicGen). The same transform generates new demos for training learned policies (DemoGen). Keep only synthesized demos that succeed in sim.
- **Numpy-only feasible:** yes. Use 4×4 homogeneous matrices or `scipy.spatial.transform.Rotation`, per-frame `np.var` over time-aligned segments, linear/slerp interpolation for transits, and sim replay as the filter.
- **Describes the skill to a caller as:** "during 'insert', motion is defined in B's frame. Invariant to B's pose within ±15 cm / ±40° of the demo range. Frame chosen because cross-demo std was 3 mm in B's frame vs 40 mm in the world frame."
- **Helps when / fails when:**
  - Helps: pick/place/insert with objects that move between episodes, few demos, and data amplification.
  - Fails: articulated or deformable objects whose geometry changes, new poses that need a different strategy (reach limits, obstacles), or two co-moving objects that make the frame ambiguous.
- **How to test it:** Compare world-frame replay, object-frame replay, and a learned policy trained on synthesized demos. Metric: success as a function of object-pose perturbation magnitude in new scenes.
- **References:**
  - Calinon, "A Tutorial on Task-Parameterized Movement Learning and Retrieval", Intelligent Service Robotics 9(1), 2016. DOI 10.1007/s11370-015-0187-9
  - Mandlekar et al., "MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations", CoRL 2023. arXiv:2310.17596
  - Garrett et al., "SkillMimicGen: Automated Demonstration Generation for Efficient Skill Learning and Deployment", CoRL 2024. arXiv:2410.18907
  - Xue et al., "DemoGen: Synthetic Demonstration Generation for Data-Efficient Visuomotor Policy Learning", RSS 2025. arXiv:2502.16932

### E.5 Task symmetries and invariances (rigid motion, permutation, object count, time-scaling)
- **Assumes:** The outcome is unchanged (or changes predictably) under global translation or planar rotation (SE(2)/SO(3) equivariance), permutation of identical objects, added irrelevant objects, and uniform change of execution speed.
- **Executable form:** (a) Canonicalise: express the state relative to a reference (target object or robot base) and rotate into a canonical heading. Map actions back afterwards. (b) Permutation: apply one shared per-object encoder, then sum/max-pool (Deep Sets), or sort identical objects by a canonical key. (c) Count: select relevant entities (relevance mask or attention) so extra objects do not change the output. (d) Time-scaling: use a phase variable s ∈ [0, 1] (the DMP canonical system) or DTW-aligned demos, so the skill runs at any speed scale τ. (e) Augment demos by applying the group actions.
- **Numpy-only feasible:** yes. Canonicalisation is matrix algebra. Deep-Sets-style: hand features per object, sum pooling, then ridge/kNN. DTW is an O(T²) numpy DP. A DMP is fitted by least squares on RBF features of the phase.
- **Describes the skill to a caller as:** "equivariant to planar rotation and translation of the A–B pair; invariant to the ordering of identical blocks; tested with 0–5 distractors; nominal duration 120 steps, valid for speed scale 0.5–2×."
- **Helps when / fails when:**
  - Helps: sample efficiency and extrapolation to unseen poses and object counts.
  - Fails: the robot breaks the symmetry (joint limits, reach, table edges), the objects are distinguishable, or the dynamics depend on speed (sliding, throwing, pouring).
- **How to test it:** Train with and without canonicalisation or augmentation. Metrics: success on poses rotated or translated beyond the demo range, and on scenes with +k distractors. For time-scaling, measure success at τ ∈ {0.5, 1, 2}.
- **References:**
  - Zaheer et al., "Deep Sets", NeurIPS 2017. arXiv:1703.06114
  - Ijspeert et al., "Dynamical Movement Primitives: Learning Attractor Models for Motor Behaviors", Neural Computation 25(2), 2013. DOI 10.1162/NECO_a_00393
  - Zhou, Kumar, Finn, Rajeswaran, "Policy Architectures for Compositional Generalization in Control", RLC 2024 (Reinforcement Learning Journal; earlier ICML 2022 workshop). arXiv:2203.05960
  - Wang et al., "Equivariant Diffusion Policy", CoRL 2024. arXiv:2407.01812

### E.6 Goal-conditioned structure and hindsight relabeling
- **Assumes:** A skill is a family π(a | s, g) indexed by a goal g. Any trajectory (demo, play or failed rollout) is a valid demonstration of how to reach the states it actually reached.
- **Executable form:** Define the goal space (target pose of an object, or a relation set from E.8). Relabel each (s_t, a_t) with goals drawn from future states s_{t+k}, the final state, or the relations actually achieved. Train goal-conditioned BC (GCSL) or a goal-conditioned value. Add failed sim rollouts under the goals they did achieve (HER).
- **Numpy-only feasible:** yes. Relabeling is index arithmetic. The policy can be kNN or locally weighted regression over (s, g) → a, or ridge regression on features of (g − s) expressed in object frames.
- **Describes the skill to a caller as:** "reaches any goal pose g = (x, y, θ) of A inside region W. Success = ‖A − g‖ < 1 cm ∧ |Δθ| < 5°. Trained on 30 demos + 2k hindsight-relabeled rollouts."
- **Helps when / fails when:**
  - Helps: sparse success signals, families of goals (place-at-x), and cheap reuse of sim data.
  - Fails: goals that are not end-states (e.g. "pour slowly"), relabeled data skewed toward easy outcomes, or goals that depend on history.
- **How to test it:** Train the same goal-conditioned learner on demos only, then on demos + relabeled rollouts. Metric: success on goal values outside the demo set.
- **References:**
  - Andrychowicz et al., "Hindsight Experience Replay", NeurIPS 2017. arXiv:1707.01495
  - Lynch et al., "Learning Latent Plans from Play", CoRL 2019. arXiv:1903.01973
  - Ghosh et al., "Learning to Reach Goals via Iterated Supervised Learning" (GCSL), ICLR 2021. arXiv:1912.06088
  - Park et al., "OGBench: Benchmarking Offline Goal-Conditioned RL", ICLR 2025. arXiv:2410.20092

### E.7 Contact-mode sequences and hybrid dynamics
- **Assumes:** Manipulation is a hybrid system. Dynamics are smooth within a contact mode (free, grasped, sticking, sliding, resting on a support) and jump at mode switches. The mode schedule is the key structural decision, and the continuous parameters are secondary.
- **Executable form:** Label each demo step with a discrete mode tuple from contacts and relative velocities, e.g. (gripper–A: none/grasp, A–table: stick/slide/lift, A–B: contact/no). Compress each demo to its mode sequence. The skill is then one controller per mode plus guard predicates for the transitions. A planner enumerates mode sequences, then solves the continuous parameters per mode (contact-mode-guided planning).
- **Numpy-only feasible:** yes, if the sim exposes contacts, or if contact can be inferred from distances (< 1 mm) and relative velocities. Mode labels come from thresholds. Per-mode linear dynamics are fitted with `lstsq`, and guard parameters are found by `scipy.optimize` or grid search in sim.
- **Describes the skill to a caller as:** "mode sequence free → grasp(A) → A-sliding-on-table → A-contacts-B. Currently in mode 2. Guard to mode 3: dist(A, B) < 1 mm ∧ normal velocity < 0."
- **Helps when / fails when:**
  - Helps: pushing, pivoting, sliding and insertion, where a single smooth policy averages across modes and fails.
  - Fails: contact signals are noisy or unavailable, many simultaneous contacts explode the mode count, or soft contacts blur the boundaries.
- **How to test it:** Compare one smooth policy with a mode-switching policy with learned guards. Metrics: success, mode-sequence match to the demos, and the fraction of failures located at mode boundaries.
- **References:**
  - Kroemer, van Hoof, Neumann, Peters, "Learning to Predict Phases of Manipulation Tasks as Hidden States", ICRA 2014. DOI 10.1109/ICRA.2014.6907441
  - Posa, Cantu, Tedrake, "A Direct Method for Trajectory Optimization of Rigid Bodies through Contact", IJRR 33(1), 2014. DOI 10.1177/0278364913506757
  - Cheng et al., "Contact Mode Guided Motion Planning for Quasidynamic Dexterous Manipulation in 3D", ICRA 2022. arXiv:2105.14431
  - Pang et al., "Global Planning for Contact-Rich Manipulation via Local Smoothing of Quasi-Dynamic Contact Models", IEEE T-RO 2023. arXiv:2206.10787

### E.8 Spatial relations and scene graphs as goals
- **Assumes:** Task goals are sets of relations between objects (on, in, left-of, aligned-with, near), invariant to exact poses. A scene graph of these relations is the right abstraction for planning and success checking.
- **Executable form:** Define each relation as a numeric function of poses and sizes. Examples: on(A, B) = xy-overlap ∧ |A.bottom − B.top| < tol ∧ A at rest; in(A, C) = A's centre inside C's box; left-of(A, B) = x_A < x_B − margin. Compute the scene graph of every demo's final state. Goal = relations true in all demo end states, minus those also true in all start states. Plan by achieving the missing relations. ReKep-style alternative: write each relation as a cost over keypoints and minimise it stage by stage.
- **Numpy-only feasible:** yes. Predicates are vectorised comparisons over pose arrays, goal extraction is set intersection across demos, and relation costs are minimised with `scipy.optimize.minimize`.
- **Describes the skill to a caller as:** "achieves {on(A, B), left-of(B, C)}. Requires clear(B) ∧ graspable(A). Success = both relations hold for 10 consecutive steps."
- **Helps when / fails when:**
  - Helps: rearrangement, stacking and sorting. Generalises across object poses and counts, and doubles as the success check.
  - Fails: goals are metric (exact pose), continuous-quality ("neat"), or outside the predicate vocabulary, or tolerances do not scale with object size.
- **How to test it:** Compare an exact-pose-matching goal with a relation-set goal. Metrics: success when objects start in new poses or new counts, and predicate false-positive/false-negative rates against sim ground truth.
- **References:**
  - Zhu et al., "Hierarchical Planning for Long-Horizon Manipulation with Geometric and Symbolic Scene Graphs", ICRA 2021. arXiv:2012.07277
  - Liu et al., "StructFormer: Learning Spatial Structure for Language-Guided Semantic Rearrangement of Novel Objects", ICRA 2022. arXiv:2110.10189
  - Migimatsu, Bohg, "Grounding Predicates through Actions", ICRA 2022. arXiv:2109.14718
  - Huang et al., "ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation", CoRL 2024. arXiv:2409.01652

### E.9 Symbolic operators: learned preconditions and effects
- **Assumes:** Each skill behaves like a STRIPS/PDDL operator. It can run when a precondition set holds, and it changes a small, predictable set of predicates (add/delete effects). Everything else stays unchanged (the frame assumption).
- **Executable form:** From predicate-labeled segment transitions (E.2 + E.8, demos + sim rollouts), for each skill: effects = predicates that consistently flip; preconditions = predicates true at every successful start, pruned or extended using sim failures. Search over the operators with BFS/A*. Bilevel variant: each operator carries a sampler for its continuous parameters (NSRTs). Invent a new predicate when outcomes cannot be separated by the existing ones.
- **Numpy-only feasible:** yes. Predicates are boolean arrays, operators come from set intersection and difference, the planner is BFS in Python, and samplers are Gaussian fits to successful parameters.
- **Describes the skill to a caller as:** "pick(A): pre = {handempty, clear(A), reachable(A)}; add = {holding(A)}; del = {handempty, on(A, ·)}. Empirical success 0.94 when pre holds, 0.05 otherwise."
- **Helps when / fails when:**
  - Helps: long-horizon composition, new goal conjunctions, and explaining skills to a planner.
  - Fails: the predicate vocabulary misses a hidden geometric condition, effects are stochastic or context-dependent, or there are side effects on other objects.
- **How to test it:** Compare a planner over learned operators with a flat policy and with hand-written operators. Metrics: success on unseen goal conjunctions, and precondition accuracy (predicted feasible vs actual sim outcome).
- **References:**
  - Konidaris, Kaelbling, Lozano-Pérez, "From Skills to Symbols: Learning Symbolic Representations for Abstract High-Level Planning", JAIR 2018. DOI 10.1613/jair.5575
  - Silver et al., "Learning Symbolic Operators for Task and Motion Planning", IROS 2021. arXiv:2103.00589
  - Chitnis et al., "Learning Neuro-Symbolic Relational Transition Models for Bilevel Planning", IROS 2022. arXiv:2105.14074
  - Silver et al., "Predicate Invention for Bilevel Planning", AAAI 2023. arXiv:2203.09634

### E.10 Affordances and cross-embodiment retargeting
- **Assumes:** Where and how to interact (contact region + post-contact motion direction) belongs to the object and is shared across agents (human → robot, robot A → robot B). The embodiment-agnostic content of a demo is the object's motion, not the hand or joint motion.
- **Executable form:** From demos (or human video), extract the contact regions in each object's frame and the object's post-contact trajectory. At test time: sample a contact from that distribution, compute an approach/grasp for this robot via IK, and track the object trajectory, retargeting object flow rather than joints. Interaction-based affordance (Where2Act): probe objects in sim with primitive pushes/pulls at sampled points, record which points move the target DoF, and fit a score map.
- **Numpy-only feasible:** partly. Without images, VRB/HOI hotspot models cannot be learned. Feasible parts: object-frame contact maps from demo state, a KDE over contact points (`scipy.stats.gaussian_kde`), sim probing, and object-trajectory retargeting through IK/sim.
- **Describes the skill to a caller as:** "affordance of drawer D: pull the handle region (D-frame box h) along +x_D. Contact distribution from 20 demos. Embodiment-independent object path 'open 25 cm'."
- **Helps when / fails when:**
  - Helps: new object instances or poses, a different gripper or robot, and human-sourced demos.
  - Fails: the new embodiment cannot achieve the contact (reach, gripper width), the affordance depends on unobserved properties (mass, locked joint), or the task is dexterous multi-contact.
- **How to test it:** Compare object-path retargeting + sampled contacts with joint-trajectory replay. Metric: success on new object instances and on a changed gripper geometry in sim.
- **References:**
  - Nagarajan, Feichtenhofer, Grauman, "Grounded Human-Object Interaction Hotspots from Video", ICCV 2019. arXiv:1812.04558
  - Mo et al., "Where2Act: From Pixels to Actions for Articulated 3D Objects", ICCV 2021. arXiv:2101.02692
  - Bahl et al., "Affordances from Human Videos as a Versatile Representation for Robotics" (VRB), CVPR 2023. arXiv:2304.08488
  - Zhu et al., "Vision-based Manipulation from Single Human Video with Open-World Object Graphs" (ORION), Autonomous Robots 2026. arXiv:2405.20321

### E.11 Tool-use structure (grasp part vs. functional part)
- **Assumes:** A tool splits into a grasp part and a functional part (effector point + effect direction). The task constrains only the functional part relative to the target. The grasp must be task-oriented (it must leave the functional motion feasible), not merely stable.
- **Executable form:** Represent the tool by keypoints: grasp point g, function point f, effect direction d (kPAM / KETO). The task becomes constraints on (f, d) relative to the target, e.g. f touches the target's far side and d points toward the goal. Solve jointly for tool pose and grasp by constrained optimisation. EE trajectory = tool trajectory ∘ grasp transform. Score grasp candidates by task rollouts in sim (task-oriented grasping).
- **Numpy-only feasible:** yes, if tool geometry is exposed as primitives or points. Keypoints come from geometry (handle centroid; the extreme point along the head axis). Constraints are solved with `scipy.optimize.minimize` (SLSQP), and candidate grasps are ranked by sim success.
- **Describes the skill to a caller as:** "tool T: grasp at handle keypoint g; hook-tip keypoint f must contact target O's far side; then pull f along −x by 20 cm. Grasp chosen so the arm stays clear of O."
- **Helps when / fails when:**
  - Helps: new tool shapes, reach-extension, and hook/push/scoop tasks.
  - Fails: the function depends on material or compliance, a tool has several functional parts, or keypoints are ill-defined for irregular shapes.
- **How to test it:** Compare "stable grasp first, then plan" with keypoint-constrained, task-oriented grasping. Metric: task success over unseen tool shapes and poses in sim.
- **References:**
  - Fang et al., "Learning Task-Oriented Grasping for Tool Manipulation from Simulated Self-Supervision", RSS 2018 (IJRR journal version). arXiv:1806.09266
  - Manuelli, Gao, Florence, Tedrake, "kPAM: KeyPoint Affordances for Category-Level Robotic Manipulation", ISRR 2019. arXiv:1903.06684
  - Qin et al., "KETO: Learning Keypoint Representations for Tool Manipulation", ICRA 2020. arXiv:1910.11977
  - Xu et al., "Creative Robot Tool Use with Large Language Models" (RoboTool), arXiv 2023. arXiv:2310.13065

### E.12 Corrective data against compounding error (DAgger family)
- **Assumes:** A policy fitted only to expert states drifts into unseen states. Robustness needs action labels for states near, but off, the demo manifold. An expert (scripted controller, planner, or local dynamics model) can supply them.
- **Executable form:**
  - DAgger: roll out the current policy in sim, label the visited states with expert actions (e.g. the object-relative waypoint tracker from E.1/E.4), aggregate, and refit.
  - DART: inject noise while replaying or collecting demos so that the expert's recoveries are recorded.
  - HG-DAgger: request labels only when a gate or error detector fires.
  - CCIL: fit local dynamics from demos and synthesise corrective (s′, a′) pairs that lead back to demo states, only where the dynamics are locally continuous.
- **Numpy-only feasible:** yes. The expert is an analytic controller, perturbed starts come from sim reset plus a state offset, and refitting uses least squares or kNN.
- **Describes the skill to a caller as:** "recovers from EE/object perturbations up to 3 cm during 'transport' (5 DAgger rounds, 1.2k labeled states). Returns to the nearest demo phase."
- **Helps when / fails when:**
  - Helps: learned (non-replay) policies, long horizons, and noisy dynamics.
  - Fails: no cheap expert exists off-distribution, perturbations leave the recoverable set (object dropped), or the dynamics are discontinuous (CCIL).
- **How to test it:** Compare plain BC, BC + DART noise, and DAgger. Metric: success under injected perturbations of increasing magnitude at random times.
- **References:**
  - Ross, Gordon, Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" (DAgger), AISTATS 2011. arXiv:1011.0686
  - Laskey et al., "DART: Noise Injection for Robust Imitation Learning", CoRL 2017. arXiv:1703.09327
  - Kelly et al., "HG-DAgger: Interactive Imitation Learning with Human Experts", ICRA 2019. arXiv:1810.02890
  - Ke et al., "CCIL: Continuity-based Data Augmentation for Corrective Imitation Learning", ICLR 2024. arXiv:2310.12972

### E.13 Runtime error detection, retry and fallback structure
- **Assumes:** Failures can be detected at runtime from state (violated precondition, stalled progress, unexpected effect). Most are recoverable by re-entering an earlier phase or switching to a fallback, rather than aborting.
- **Executable form:** Wrap each skill as a behaviour tree: `Sequence(check_pre, run, check_post)` under `Fallback(retry ≤ k from the last satisfied phase, alternative skill, return failure code)`. Monitors:
  - progress: distance-to-subgoal not decreasing for W steps;
  - invariant violation: e.g. holding(A) becomes false during transport;
  - action inconsistency / erratic behaviour;
  - timeout.

  Return a structured failure summary (which check failed, and when) so the caller or planner can react (REFLECT-style). Recovery-RL-style: a separate recovery policy takes over when predicted constraint violation is high.
- **Numpy-only feasible:** yes. Monitors are predicates and moving-window statistics on state, and the behaviour tree is plain Python. Validate by forcing failures in sim (e.g. moving an object mid-skill).
- **Describes the skill to a caller as:** "retries the grasp up to 3× if holding(A) is false after closing. Raises 'A_unreachable' if pre fails. Flags 'stalled' after 50 steps without progress. Cannot recover if A leaves the table."
- **Helps when / fails when:**
  - Helps: stochastic contacts, grasp misses, and long chains where one failure ends the episode.
  - Fails: failures are irreversible, false alarms trigger pointless retries, or retries loop without changing anything.
- **How to test it:** Compare skills with and without monitors + retry under injected disturbances. Metrics: episode success, mean time, and detector precision/recall against sim-ground-truth failures.
- **References:**
  - Colledanchise, Ögren, "Behavior Trees in Robotics and AI: An Introduction", CRC Press 2018. arXiv:1709.00084
  - Thananjeyan et al., "Recovery RL: Safe Reinforcement Learning with Learned Recovery Zones", RA-L 2021. arXiv:2010.15920
  - Liu et al., "REFLECT: Summarizing Robot Experiences for Failure Explanation and Correction", CoRL 2023. arXiv:2306.15724
  - Agia et al., "Unpacking Failure Modes of Generative Policies: Runtime Monitoring of Consistency and Progress", CoRL 2024. arXiv:2410.04640

### E.14 Skill chaining and reusable skill libraries
- **Assumes:** Complex tasks are compositions of reusable skills, each with an explicit initiation set and termination set. Composition works when each skill's termination distribution lies inside the next skill's initiation set. Skills learned for one task transfer to other tasks on the same robot.
- **Executable form:** Store each skill as {policy/code, initiation classifier, termination predicate, parameters, stats, provenance}.
  - Skill chaining: learn skills backward from the goal. Skill k's initiation set becomes the target of skill k−1.
  - Terminal-state regularisation: push skill k−1's end states toward the region where skill k succeeds.
  - Lifelong libraries: store skills as executable code retrieved by description (Voyager). Add a new skill only when a demo segment matches no existing skill (LOTUS).
- **Numpy-only feasible:** yes. Initiation sets are classifiers (logistic regression, kNN, or a Mahalanobis ellipsoid) fitted on sim rollout outcomes from sampled start states. The library is a Python module plus JSON metadata.
- **Describes the skill to a caller as:** "place_on(A, B): initiation = holding(A) ∧ reachable(B) (97% success inside, 12% outside, n=300); terminates in on(A, B) with gripper open. Used in tasks T1 and T3."
- **Helps when / fails when:**
  - Helps: long horizons, and many related tasks on one robot. Hand-off failures become explicit and measurable.
  - Fails: an upstream skill's end states drift outside the downstream initiation set, the library bloats with near-duplicates, or retrieval returns a subtly wrong skill.
- **How to test it:** Compare chaining with and without initiation/termination alignment. Metrics: per-hand-off success and full-task success. For reuse: sim steps (or agent iterations) to solve a new task with vs without the library.
- **References:**
  - Konidaris, Barto, "Skill Discovery in Continuous Reinforcement Learning Domains using Skill Chaining", NeurIPS (NIPS) 2009.
  - Lee, Lim, Anandkumar, Zhu, "Adversarial Skill Chaining for Long-Horizon Robot Manipulation via Terminal State Regularization", CoRL 2021. arXiv:2111.07999
  - Wang et al., "Voyager: An Open-Ended Embodied Agent with Large Language Models", TMLR 2024. arXiv:2305.16291
  - Wan et al., "LOTUS: Continual Imitation Learning for Robot Manipulation Through Unsupervised Skill Discovery", ICRA 2024. arXiv:2311.02058

### E.15 Reward inference from demonstrations and preferences (IRL)
- **Assumes:** Demonstrators (near-)optimise an unknown reward that is simple in the right features. That reward transfers across scenes better than the demonstrated motion does. Pairwise preferences between trajectories reveal the same reward.
- **Executable form:** Choose features φ(s, a): distances, relation indicators, tilt, contact, time.
  - MaxEnt IRL: fit weights w so that feature expectations under the soft-optimal policy match the demos.
  - Preferences: Bradley–Terry logistic regression on φ(τ_A) − φ(τ_B), combining demos and preferences (Bıyık).
  - LLM-written reward (Eureka): write the reward as code, evaluate it through policy rollouts, and revise.

  Then optimise a policy for the learned reward in sim (CEM/MPC).
- **Numpy-only feasible:** partly. Linear-feature IRL and Bradley–Terry are small `scipy.optimize` problems. The inner optimisation needs many sim rollouts. CEM in numpy is feasible for short horizons and costly for long ones.
- **Describes the skill to a caller as:** "optimises r = −2.1·dist(A, goal) − 0.3·tilt(A) − 0.05·t (from 20 demos). Ranks demos above perturbed demos 96% of the time."
- **Helps when / fails when:**
  - Helps: the goal transfers but the motion does not (new layouts), and ranking candidate policies.
  - Fails: rewards are ambiguous (many explain the demos), key features are missing, the demonstrator is systematically suboptimal, or the policy hacks the reward in sim.
- **How to test it:** Compare BC with optimising the inferred reward on scenes whose layout differs from the demos. Metrics: true success, and ranking accuracy of the reward (demo vs perturbed trajectories).
- **References:**
  - Ziebart, Maas, Bagnell, Dey, "Maximum Entropy Inverse Reinforcement Learning", AAAI 2008.
  - Christiano et al., "Deep Reinforcement Learning from Human Preferences", NeurIPS 2017. arXiv:1706.03741
  - Bıyık et al., "Learning Reward Functions from Diverse Sources of Human Feedback: Optimally Integrating Demonstrations and Preferences", IJRR 2022. arXiv:2006.14091
  - Ma et al., "Eureka: Human-Level Reward Design via Coding Large Language Models", ICLR 2024. arXiv:2310.12931

### E.16 Constraint inference from demonstrations
- **Assumes:** Demos satisfy hard constraints (keep upright, stay above the table, avoid region R, keep A aligned with B's axis). These show up as what demonstrators consistently keep invariant or never do. Constraints transfer across tasks more readily than rewards.
- **Executable form:**
  - Invariance-based (C-LEARN): a feature with near-zero variance across demos in a phase becomes an equality constraint, and a feature's demo range, with margin, becomes an inequality bound.
  - Avoidance-based (maximum-likelihood constraint inference): states that are cheaper to pass through but never visited are likely infeasible. Greedily add the candidate constraint that most increases demo likelihood, using nominal (unconstrained) sim rollouts as the contrast.
  - Shared constraints: pool demos across tasks.

  Enforce the constraints through constrained optimisation or projection, and reuse them as runtime monitors (E.13).
- **Numpy-only feasible:** yes. Use per-feature variance and percentiles, greedy selection over a candidate constraint library with sim rollouts, and `scipy.optimize.minimize(method="SLSQP", constraints=…)`.
- **Describes the skill to a caller as:** "during 'transport' keeps tilt(cup) < 8° (max tilt across demos: 3°) and never enters region R (0/50 demos, vs 34% of unconstrained rollouts)."
- **Helps when / fails when:**
  - Helps: safety and manner-of-execution requirements (carrying liquids, clearances), where the goal alone underspecifies the task.
  - Fails: with few demos, incidental regularities become spurious constraints, and over-tight constraints make new scenes infeasible.
- **How to test it:** Execute with and without the inferred constraints. Metrics: ground-truth violation rate, success in new scenes, and the spurious-constraint rate as a function of demo count.
- **References:**
  - Pérez-D'Arpino, Shah, "C-LEARN: Learning Geometric Constraints from Demonstrations for Multi-Step Manipulation in Shared Autonomy", ICRA 2017. DOI 10.1109/ICRA.2017.7989466
  - Chou, Berenson, Ozay, "Learning Constraints from Demonstrations", WAFR 2018. arXiv:1812.07084
  - Scobee, Sastry, "Maximum Likelihood Constraint Inference for Inverse Reinforcement Learning", ICLR 2020. arXiv:1909.05477
  - Kim et al., "Learning Shared Safety Constraints from Multi-task Demonstrations", NeurIPS 2023. arXiv:2309.00711

### E.17 Temporal task specifications (LTL / reward machines)
- **Assumes:** Task success is a temporal pattern over predicates (eventually X, X before Y, always not Z) that a small automaton can represent. The automaton state is the task's progress, so the task is non-Markovian in the raw state but Markovian in (state, automaton state).
- **Executable form:** Enumerate formula templates over the E.8 predicates: ordering, avoidance, conjunction of eventualities. Keep the formulas satisfied by all demos and violated by negatives (random or failed rollouts), preferring short formulas (Bayesian specification inference). Compile the result into a reward machine or finite automaton. The high-level policy selects skills by automaton state and tracks progress by stepping the automaton. An LLM can translate a language instruction into a formula that is then checked (AutoTAMP).
- **Numpy-only feasible:** yes. Predicate traces are boolean arrays (T × P), template checks are vectorised (`np.argmax` for first-true indices), and the automaton is a dict.
- **Describes the skill to a caller as:** "spec: F on(A, B) ∧ G ¬collide(C) ∧ (grasp(A) before on(A, B)). Automaton state 2 of 3. Violation of G ¬collide(C) → failure."
- **Helps when / fails when:**
  - Helps: ordered multi-step tasks, non-Markovian goals, and checkable, explainable success.
  - Fails: the predicates miss what matters, template enumeration explodes, or too few demos to tell required ordering from incidental ordering.
- **How to test it:** Compare an end-state goal/reward with an automaton-structured reward and skill selection. Metrics: success on tasks with required orderings, and inferred-spec accuracy against ground truth as the demo count varies.
- **References:**
  - Toro Icarte, Klassen, Valenzano, McIlraith, "Using Reward Machines for High-Level Task Specification and Decomposition in Reinforcement Learning", ICML 2018. https://proceedings.mlr.press/v80/icarte18a.html
  - Vazquez-Chanlatte et al., "Learning Task Specifications from Demonstrations", NeurIPS 2018. arXiv:1710.03875
  - Shah, Kamath, Li, Shah, "Bayesian Inference of Temporal Task Specifications from Demonstrations", NeurIPS 2018.
  - Chen et al., "AutoTAMP: Autoregressive Task and Motion Planning with LLMs as Translators and Checkers", ICRA 2024. arXiv:2306.06531

### E.18 Language as a structuring prior
- **Assumes:** Natural language supplies a vocabulary of reusable subtasks and parameters. Language-level decomposition matches the task's real segmentation. Skills indexed by short descriptions compose through LLM planning, provided each step is grounded by a feasibility estimate.
- **Executable form:** Name each segmented phase (E.2) with a verb–object phrase ("grasp red block"). Expose each skill as a Python function with a docstring and typed parameters (Code-as-Policies). The planner (the coding agent itself) chains the named skills, scoring each step by usefulness × feasibility, where feasibility is the initiation classifier from E.14 (SayCan). Keep an intermediate layer of language motions ("move gripper left of A") that allows correction (RT-H).
- **Numpy-only feasible:** yes. Language here is metadata and API design. No language model is needed inside the policy.
- **Describes the skill to a caller as:** "`put(obj, target)`: 'places the held object on target; requires holding(obj)'. Composes after `pick(obj)`."
- **Helps when / fails when:**
  - Helps: new instruction compositions, interpretability, and human or agent correction.
  - Fails: the name promises more than the skill does, instructions are ambiguous, or language granularity does not match control granularity.
- **How to test it:** Give the planner named skills with docstrings, then the same skills as opaque IDs. Metrics: success on new instruction compositions and planning accuracy.
- **References:**
  - Ahn et al., "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances" (SayCan), CoRL 2022. arXiv:2204.01691
  - Sharma, Torralba, Andreas, "Skill Induction and Planning with Latent Language", ACL 2022. arXiv:2110.01517
  - Liang et al., "Code as Policies: Language Model Programs for Embodied Control", ICRA 2023. arXiv:2209.07753
  - Belkhale et al., "RT-H: Action Hierarchies Using Language", RSS 2024. arXiv:2403.01823

### E.19 Success detectors and self-verification
- **Assumes:** Success is easier to recognise than to produce. A detector fitted to demo end states generalises, and can gate retries, supply rewards for sim practice, and verify each skill before its result is reported.
- **Executable form:** Classifier over state features. Positives = the last k states of successful demos and sim successes. Negatives = earlier demo states and failed rollouts. VICE-style: iteratively add states the policy reaches as negatives, so the detector is not exploited. Self-verification loop (Inner Monologue / Voyager): after each skill, evaluate the post-condition and feed the verdict back to the planner. Learned VLM failure detectors (AHA) are the image-domain analogue.
- **Numpy-only feasible:** yes. Options: a logistic regression or Gaussian density on features, or explicit predicate thresholds taken from the demo end-state distribution (min/max plus a margin). Evaluate against sim ground truth on fresh seeds.
- **Describes the skill to a caller as:** "success(s) = ¬holding(A) ∧ ‖A − B_top‖ < 1.2 cm ∧ ‖v_A‖ < 1e-3. Precision 0.98 / recall 0.95 on 200 held-out sim rollouts."
- **Helps when / fails when:**
  - Helps: sparse rewards, autonomous practice in sim, and retry/abort decisions.
  - Fails: the policy exploits the detector (false positives in unseen states), success depends on hidden properties, or an end-state-only check misses process failures (e.g. spills).
- **How to test it:** Measure detector agreement with sim ground truth. Then train or gate the policy with the learned detector vs the ground-truth signal. Metrics: precision/recall and downstream task success.
- **References:**
  - Fu, Singh, Ghosh, Yang, Levine, "Variational Inverse Control with Events: A General Framework for Data-Driven Reward Definition" (VICE), NeurIPS 2018. arXiv:1805.11686
  - Huang et al., "Inner Monologue: Embodied Reasoning through Planning with Language Models", CoRL 2022. arXiv:2207.05608
  - Du et al., "Vision-Language Models as Success Detectors", CoLLAs 2023. arXiv:2303.07280
  - Duan et al., "AHA: A Vision-Language-Model for Detecting and Reasoning Over Failures in Robotic Manipulation", ICLR 2025. arXiv:2410.00371

### E.20 Calibrated and conformal confidence of skills
- **Assumes:** A skill's reliability should be reported as a calibrated probability, or as a distribution-free bound estimated on held-out episodes from the same distribution. Low confidence should trigger asking for help or a fallback, not silent execution.
- **Executable form:**
  - Success-rate bounds: run N fresh-seed sim episodes from the stated operating region and report a Clopper–Pearson or Beta-posterior lower bound.
  - Split conformal runtime monitor: calibrate a nonconformity score (kNN distance to demo states, ensemble disagreement, or detector margin) on held-out successful runs, and raise an alarm when the score exceeds the (1−α)(n+1)/n quantile (FAIL-Detect).
  - Choice under uncertainty (KnowNo): form a prediction set over candidate plans or skills, and ask for help or disambiguation when it has more than one element.
- **Numpy-only feasible:** yes. `scipy.stats.beta.ppf` for bounds, `np.quantile` for conformal thresholds, and `scipy.spatial.cKDTree` for kNN scores.
- **Describes the skill to a caller as:** "success 0.91 (95% lower bound 0.86, n=200) for A in region R. Uncalibrated outside R. Runtime alarm when the score exceeds the calibrated 95% quantile (target false-alarm rate 5%)."
- **Helps when / fails when:**
  - Helps: deciding whether to call a skill, comparing alternative skills, and triggering fallback or help.
  - Fails: deployment breaks exchangeability (calibration scenes not representative), small n gives vacuous bounds, or the score does not predict failure.
- **How to test it:** Check empirical coverage and calibration curves on new sim scenes. Then compare with and without confidence gating. Metrics: failure rate among accepted calls, and false-alarm rate.
- **References:**
  - Angelopoulos, Bates, "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification", 2021. arXiv:2107.07511
  - Ren et al., "Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners" (KnowNo), CoRL 2023. arXiv:2307.01928
  - Vincent et al., "How Generalizable Is My Behavior Cloning Policy? A Statistical Approach to Trustworthy Performance Evaluation", IEEE RA-L 2024. arXiv:2405.05439
  - Xu et al., "Can We Detect Failures Without Failure Data? Uncertainty-Aware Runtime Failure Detection for Imitation Learning Policies" (FAIL-Detect), RSS 2025. arXiv:2503.08558
