## Prior catalog and prior cards

The priors are the policy's interface: the high-level agent decides whether, where and how to call a policy from its prior description. The prior catalog (read_catalog: an empty list returns its index, IDs return those entries) lists about a hundred priors in five families: A representation and perception, B objective and data, C architecture, D control and planning, E task and demonstration structure. Each entry states what it assumes, how to implement it, what it tells a caller, when it helps or fails, how to test it, and literature. Read its index before designing the prior set and open the entries you consider. Its notes about a numpy-only sandbox do not apply here: in this framework a prior acts through the policy's inputs, its Cartesian action representation and frame, encoders or conditioning, label-preserving data transformations, auxiliary objectives or the declared HL arguments, inside the fixed Diffusion Policy training.

For every prior in the set:
- in its statement, name the catalog entries it instantiates by ID (for example A.2 or E.5) and how each is instantiated here; a prior that is not in the catalog is welcome when described the same way;
- in its applicability, give its grounding: which objects, frames or quantities in the observation fill each role the prior needs, and how the high-level agent supplies or identifies them;
- in parameter_semantics, give each argument's physical meaning, unit and the range the demonstrations cover;
- in its limitations, use the catalog entry's failure conditions to state where the prior should break, and which other prior in the set covers that case.

Choose priors whose catalog entries differ where their targeted failures differ.
