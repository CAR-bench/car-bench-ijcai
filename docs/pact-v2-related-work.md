# Related-work survey: contract-guided, evidence-bound LLM tool agents

## Research question

How can a tool-using LLM agent generalize across natural-language requests and
live tool inventories while keeping tool execution, completion claims, and
inference compute subject to a small deterministic verifier?

The intended contribution is not another tool-routing prompt. It is the
intersection of typed plan compilation, runtime policy enforcement,
evidence-provenance checking, selective abstention, and explicit compute bounds.

## Scope and survey axes

1. **Reasoning and tool planning:** model-driven interleaved action, classical
   planning, and compiler/DAG approaches.
2. **Runtime assurance:** external enforcement around an untrusted agent,
   including policy DSLs and safety monitors.
3. **Evidence and executable contracts:** schema validation, correlated tool
   outcomes, and grounded completion.
4. **Selective action:** deciding when a deterministic runtime owns the next
   transition and when it must abstain or recompile.

Training new foundation models, generic content moderation, and hidden-state
inspection are outside scope.

## Taxonomy

### A. Neural action generation

- ReAct interleaves language reasoning and environmental actions. It offers
  flexibility but places every transition inside stochastic generation.
- Multi-agent tool frameworks add specialized selector, executor, and critic
  roles, usually increasing calls and weakening the small trusted boundary.

### B. LLM-to-plan compilation

- LLM+P translates natural language into PDDL and delegates search to a
  classical planner.
- LLMCompiler generates a dependency graph for efficient function execution,
  emphasizing parallelism, latency, and cost.
- PlanCompiler generates a typed JSON plan over a fixed primitive registry,
  applies static graph/type validation, and executes the compiled pipeline
  deterministically. It is the closest architectural prior work.
- PACT-V2 similarly uses the model as a compiler, but its IR is an online
  obligation/evidence graph over a changing A2A tool contract. It executes one
  benchmark-visible action at a time and prioritizes runtime validity over
  parallel speedup.

### C. Runtime policy enforcement

- Simplex architectures supervise an unverifiable advanced controller with a
  small assurance mechanism and safe fallback.
- AgentSpec provides a domain-specific language for triggers, predicates, and
  runtime enforcement of LLM-agent constraints.
- GuardAgent translates textual guard requests into executable guardrail code,
  using an LLM-based guard agent.
- PACT-V2 is closest to AgentSpec in its deterministic enforcement boundary.
  Its proposed difference is to combine policy obligations with the task plan,
  correlated execution evidence, completion semantics, and a competition-level
  sequential compute ledger.

### D. Proof- and certificate-carrying execution

- Proof-Carrying Code requires untrusted code to ship a proof checked against a
  host safety policy.
- PACT-V2 borrows the producer/checker separation, but its action certificate
  is not a theorem-prover proof. It is a machine-checkable receipt over PlanIR,
  live schemas, dependency states, and evidence IDs. Claims must use
  “certificate-carrying action” or “proof-carrying inspiration,” not formal PCC.

### E. Agent risk and abstention

- ToolEmu demonstrates long-tail risk in tool agents and motivates systematic
  failure injection.
- CAR-bench targets consistency and limit awareness under missing capabilities,
  ambiguity, and policy.
- Selective-classification work formalizes empirical risk at a chosen coverage.
- AgentAbstain (2026) treats not acting as a sequential agent capability and
  reports that it remains difficult even for frontier systems.
- PACT-V2 abstains when its typed plan or evidence certificate is undefined;
  the trigger is verifier coverage, not raw model confidence.

## Key papers

| Work | Axis | Key contribution | Gap relative to PACT-V2 |
|---|---|---|---|
| [ReAct (ICLR 2023)](https://openreview.net/forum?id=WE_vluYUL-X) | A | Interleaved model reasoning and acting | No small deterministic plan/evidence kernel |
| [LLM+P (2023)](https://arxiv.org/abs/2304.11477) | B | LLM compiles natural language to PDDL; classical planner solves | Assumes a planning domain; not live A2A schemas or evidence-grounded completion |
| [LLMCompiler (ICML 2024)](https://proceedings.mlr.press/v235/kim24y.html) | B | Function-call DAG and parallel execution | Optimizes orchestration; does not enforce policy/evidence obligations |
| [PlanCompiler (2026)](https://arxiv.org/abs/2604.13092) | B/C | Typed plan registry, static graph validation, deterministic compilation | Fixed data-pipeline registry; no revocable live tool contract, conversational obligation state, or correlated outcome/completion semantics |
| [AgentSpec (ICSE 2026)](https://doi.org/10.1145/3744916.3764546) | C | DSL for customizable runtime constraints | Enforces guard rules, not an evidence-bound task/goal graph and completion contract |
| [GuardAgent (ICML 2025)](https://openreview.net/forum?id=2nBcjCZrrP) | C | LLM creates and executes guardrail code | Larger learned guard boundary; no schema/evidence certificate kernel |
| [Simplex (ACC 1998)](https://doi.org/10.1109/ACC.1998.703255) | C | Runtime assurance around an unverified complex controller | Continuous-control safety setting; PACT uses abstention/recompile as fallback |
| [Proof-Carrying Code (POPL 1997)](https://doi.org/10.1145/263699.263712) | D | Untrusted producer supplies proof checked by host | PACT certificates are structural receipts, not logical proofs |
| [ToolEmu (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/7274ed909a312d4d869cc328ad1c5f04-Abstract-Conference.html) | E | Scalable risk discovery for tool agents | Evaluation framework rather than runtime prevention |
| [Selective Classification (NeurIPS 2017)](https://proceedings.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html) | E | Coverage--risk formulation for abstaining predictors | Static prediction; PACT applies coverage to sequential plan execution |
| [AgentAbstain (2026)](https://arxiv.org/abs/2607.10059) | E | Paired evaluation of sequential act/abstain behavior | Benchmark rather than a typed assurance runtime |
| [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) | C/D | Measures constrained-decoding coverage and quality over 10k schemas | Motivates local validation; does not address policy or outcome provenance |
| [CAR-bench (ACL 2026)](https://aclanthology.org/2026.acl-long.1886/) | E | Consistency and limit-awareness benchmark | Evaluation target rather than the proposed mechanism |

## Identified gap

The individual ingredients are not new:

- LLMCompiler already treats function orchestration as compilation;
- PlanCompiler already uses a typed plan, static validator, and deterministic
  executor, so none of those components alone is a novelty claim;
- LLM+P already delegates model-produced structure to a deterministic solver;
- AgentSpec already enforces declarative agent constraints at runtime;
- provenance systems already link outputs to sources; and
- selective prediction already studies risk versus coverage.

The defensible gap is narrower:

> Existing work does not jointly use a bounded LLM compiler to produce an
> online obligation graph over a live, revocable tool contract; correlate every
> state transition with successful A2A execution evidence; restrict completion
> to critical evidence-satisfied goals; and expose the resulting selective
> coverage and sequential inference cost at the protocol boundary.

This gap must be validated empirically. Without the evidence ledger and PlanIR
runtime enabled in the submitted image, it is only a design direction.

## Recommended positioning

Use the phrase **contract-guided, evidence-bound selective execution**.

Avoid claiming a generic formal proof of policy compliance. The language model
still performs semantic compilation of natural-language policy. Instead,
separate guarantees:

- unconditional after IR acceptance: live capability/schema confinement,
  result correlation, failure non-escalation, exactly-once terminal emission,
  context isolation, and completion-call bound;
- conditional on compiler correctness: semantic completeness of policy and
  goal obligations; and
- empirical: reward, consistency, paraphrase robustness, verifier coverage,
  repair rate, and token use.

The clearest paper contrast is:

> Unlike ReAct, the model proposes a typed plan rather than every action.
> Unlike LLMCompiler, the graph is verified for policy/evidence obligations
> rather than optimized for parallel dispatch. Unlike AgentSpec, enforcement is
> tied to task completion and correlated tool outcomes. PACT-V2 therefore uses
> an untrusted neural compiler with a small deterministic execution kernel and
> explicit abstention boundary.
