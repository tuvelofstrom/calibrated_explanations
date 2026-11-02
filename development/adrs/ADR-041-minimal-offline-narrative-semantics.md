# ADR-041: Minimal Offline Narrative Semantics

Status: Accepted
Date: 2026-11-02
Deciders: Core maintainers
Supersedes: None
Related: ADR-001, ADR-002, ADR-005, ADR-008, ADR-020, ADR-021, ADR-029, ADR-040, ADR-042

## Context

Calibrated Explanations already renders offline template narratives through its public explanation interfaces. Downstream consumers also need an unambiguous representation of the prediction, uncertainty and explanation facts that underpin these narratives. Reconstructing those facts from presentation text or PlotSpec is error-prone.

## Decision

1. **Established CE semantics are authoritative.** The explanation payload (ADR-005) and documented explanation domain model (ADR-008) supply source facts. Narrative representation MUST NOT call a model again, recompute calibration, infer causal effects or treat PlotSpec as a semantic source.
2. **Passive representation only.** A minimal immutable, versioned `NarrativeSpec` may carry source-bound facts, already-established class/threshold/reject context, ordering, uncertainty qualifiers and conservative claim permissions. It does not define a provider protocol or an alternative generation engine. A canonical representation of a fixed input MUST be deterministic; any identifier derived from it MUST be content-based. Missing required facts MUST fail explicitly, not be synthesized.
3. **Extension within the established payload.** `metadata.narrative_context` is an optional additive ADR-005 v1 field containing only presentation-neutral, already-known task/class/threshold/reject context. Existing fields and their meanings remain unchanged. Absence MUST remain compatible with older payloads. No sensitive values are introduced without a legitimate underlying explanation source.
4. **Preserve uncertainty.** Do not round or silently coerce numerical values, finite interval bounds or explicitly unbounded endpoints. Validate applicable inclusive interval invariants per ADR-021. Attach stable source references to facts. Use canonical rule order per ADR-042; preserve explanation type and sign/direction semantics.
5. **Local offline rendering remains the default.** Preserve the existing `NarrativeGenerator`, packaged templates and public `to_narrative()`, `narrate()`, `to_dataframe()`, and `plot(style="narrative")` APIs and existing output types. No new runtime dependencies, plugin family, vendor transport, credential handling or network calls are introduced. Internal refactoring must be justified by existing functionality and regression-tested.
6. **No inflated correctness claim.** An accurate numerical source does not prove that arbitrary prose is factually or statistically correct. Template and narrative tests assess only their stated conditions; do not label prose itself calibrated. Do not introduce unverifiable guarantees or global summaries without an established semantic source.

## Consequences

This decision creates a small data boundary, not a runtime extension mechanism. The library is fully functional offline with no external narrator. Existing serialized data and narration APIs remain supported, and any intentional change to human-readable wording must be documented and tested.

## Verification

- Public lifecycle and narrative compatibility tests cover factual, alternatives and FAST where supported.
- Round-trip tests with and without `metadata.narrative_context`; deterministic canonical facts for fixed inputs.
- Invalid finite intervals, missing required context and rejected/deferred cases are exercised.
- Offline import/test smoke with network prohibited and no optional narrative provider.
- Numerical/parity gates remain unchanged.