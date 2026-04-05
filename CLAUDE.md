Experiment Summary
Core question: How does debate protocol performance degrade as a function of (a) capability gap magnitude and (b) judge verification budget, and do these interact?
Setup: A debate protocol over questions about fictional world histories, where:
* The debater models have access to the world document; the judge does not
* Knowledge asymmetry is real by construction — all models start from zero on fictional content
* Questions require both contingent facts (irreducibly empirical) and reasoning from those facts, so the task is not reducible to logical proof verification
Models: Llama 3 Instruct 1B, 3B, 8B, 70B for judges, Llama 3 Instruct 70B for debaters and oracle
Independent variables:
* Capability gap: size ratio between debater and judge models
* Query budget: number of oracle queries available to the judge (0, 1, 2, 5, unlimited)
Dependent variables:
* Judge accuracy (does the correct answer win?)
* Dishonest-debater success rate (the alignment-relevant failure mode)
* Query efficiency (does a small budget recover most of the accuracy gain?)
Key theoretical motivation: The Irving et al. debate guarantee assumes judges can evaluate argument quality without domain knowledge. This experiment tests how much explicit verification capacity is needed to compensate when that assumption fails — operationalizing a known gap between debate theory and realistic oversight conditions.

Code style and constraints
* Write in Python; we'll use uv for environment management and ty for typechecking
* I'll need to run all these models remotely and pay for compute. Don't run anything that costs money without permission.
