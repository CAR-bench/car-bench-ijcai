# Track 2 technical-report outline: final PACT submission

Use the IJCAI author kit and keep the main text to four pages, excluding
references. This outline describes the single submitted PACT runtime. Older
rule-controller and planner/executor variants are archived implementation
history and must not be presented as alternate final methods. Replace every
bracketed result with reproducible evidence from the exact release image; do
not claim hidden-set results.

## Title, authors, and abstract

- **Title:** PACT: Contract-Verified Obligation Planning for Reliable
  Tool-Using Agents
- **Authors and affiliations:** [team members and institutions]
- **Abstract:** State the Track 2 direct-Cerebras setup and the core separation:
  an untrusted semantic compiler proposes plans, while a deterministic kernel
  owns execution authority, live-contract validation, scoped confirmation, and
  completion evidence. Report only final deterministic checks and clearly
  labeled public validation.

## 1. Problem statement and trust boundary

Define the objective as reliable tool use under ambiguous goals, policy
constraints, mutable capabilities, and fallible model output. Explain that PACT
does not formally prove natural-language understanding. Its enforceable claims
begin after model output and cover strict decoding, typed PlanIR structure,
Draft 2020-12 argument validation, capability-digest freshness, evidence
provenance, exact confirmation scope, and deterministic runtime transitions.

State the anti-specialization property precisely: the reachable release kernel
contains no benchmark task IDs, memorized utterances, operation-specific
allowlists, or domain workflow branches. Semantics arrive only through the trusted policy,
current event, live capability contracts, and external evidence.

## 2. Method

Describe the following components.

1. **Untrusted semantic compiler.** A Cerebras-hosted `gpt-oss-120b` model
   returns a compact strict-schema wire plan. The submitted defaults are
   `medium` reasoning, 8,192 maximum completion tokens, and semantic review
   enabled.
2. **Proposal, repair, action audit.** PACT makes one proposal call, at most one
   value-redacted verification-guided repair, and, only for an accepted plan
   containing an Act, one independent semantic action-audit call. The audit
   returns a complete replacement candidate that is decoded and verified from
   scratch. Failure never falls back to the pre-audit plan. The maximum is
   three sequential model completions, below the Track 2 limit of five.
3. **Typed PlanIR and local verifier.** The non-executable DAG contains only
   Observe, Ask, Confirm, Act, and Respond nodes. The verifier checks graph
   bounds, live operation membership, full argument schemas, capability
   digests, confirmation prerequisites, and evidence provenance before the
   runtime can emit anything.
4. **Deterministic obligation runtime.** Stable topological execution exposes
   at most one pending external obligation. Results must match the pending
   context, plan, node, operation, arguments, schema digest, and explicit
   success status before they create evidence.
5. **Control/data event split.** Exact confirmation and successful Act events
   advance the same immutable verified suffix. Ask answers and Observe results
   introduce semantic data and trigger a new compilation horizon. Capability
   changes and qualified confirmation revisions invalidate the old horizon.
6. **Scoped confirmation contract.** A trusted capability is
   confirmation-required only through the explicit extension or an anchored
   `REQUIRES_CONFIRMATION` marker at the beginning of its trusted function
   description. Authorization is one-shot and binds schema digest, operation,
   and canonical argument digest.
7. **Exact completion closure.** Before PlanIR construction, local lowering
   replaces a `completed` response's model-authored evidence list with exactly
   all current Act success keys plus all carried successful-Act obligations.
   Only successful external Act records can discharge that proof.

Present the semantic action audit as a stochastic cross-check, not a formal
proof. The formal claims belong to the local structural, contract, evidence,
and transition checks.

## 3. Architecture and compute audit

Include one architecture diagram with this path:

```text
A2A policy / goal / live capabilities / result
                 |
                 v
       untrusted semantic compiler
       proposal -> [repair] -> [Act audit]
                 |
      strict decode + typed PlanIR + verifier
                 |
       deterministic obligation runtime
         |                         |
  one external call         evidence-grounded text
         |                         |
         +---- append-only ledger--+
                    |
          aggregate A2A turn_metrics
```

Use brackets to show optional calls and label the worst semantic path as three
sequential completions. Provider 429 retry/wait is a separately bounded
transport loop and does not accept additional semantic candidates.

Explain disjoint token accounting. Cerebras includes reasoning inside provider
output tokens, so PACT reports:

```text
prompt_tokens     = input_tokens
thinking_tokens   = reasoning_output_tokens
completion_tokens = max(0, output_tokens - reasoning_output_tokens)
```

Aggregate proposal, rejected repair, and audit usage across the external-action
loop until the next text response. Do not sum provider output and reasoning a
second time.

## 4. Validation, limitations, and reproducibility

Report only reproducible final-configuration evidence:

| Evidence | Exact configuration | Result |
| --- | --- | --- |
| Deterministic suite | final commit; unit tests, Ruff, mypy | [fill] |
| Local or Docker smoke | public/train task IDs and trial count | [fill] |
| README Option C | exact public `linux/amd64` GHCR digest | [fill] |
| Public batch, if completed | public split and trials | [fill] |

For any task run, separate inference latency from quota-wait time and source all
token totals from A2A `turn_metrics.prompt_tokens`, `completion_tokens`, and
`thinking_tokens`. A single smoke is a release-gate check, not a performance
estimate. Do not reuse scores from archived agents, pre-audit PACT builds, or
different inference defaults.

Discuss these limitations without weakening the contribution:

- natural-language semantic correctness and the model action audit are not
  formally proved;
- unknown effect metadata limits Observe-versus-Act guarantees;
- exact result origin relies on the A2A single-pending contract because no
  agent-selected nonce is echoed;
- the ledger is process-local rather than durable; and
- bounded repair/audit improves predictability but can fail closed on difficult
  requests.

Close with the immutable public GHCR digest, source commit, required
environment-variable names, hidden-set `scenario.toml`, and the exact Option C
command. Include no secrets.

## References

Include the required CAR-bench citation from the project README, plus primary
sources for the model/provider, JSON Schema, and any framework claims used in
the final report.
