# ADR 0006: Synthetic data only, independent of the sibling repositories

**Status:** accepted (2026-10-13)

## Context

The project sits conceptually downstream of `corebank-ledger` and `txn-data-pipeline`.
Real transaction data cannot be used in a public portfolio and would raise privacy and
licensing questions; depending on the sibling repositories at runtime would make the
project hard to run and would couple three codebases.

## Decision

* All data comes from a seeded generator (`fraud_ml.data.synthetic`) that produces
  legitimate behaviour and ten stylised fraud-like patterns with deliberate overlap, and
  derives history fields through the same point-in-time routine used for external data.
* Identifiers use obvious synthetic prefixes; fraud types are generator conventions and
  are documented as such.
* The repository defines its own versioned contract (`fraud-transaction/1`), ships small
  sample files, and provides an optional adapter for the curated output of
  `txn-data-pipeline`. The adapter is tested against a curated-shaped fixture; nothing
  imports the pipeline.
* Published metrics come only from runs of the committed code on generated data.

## Consequences

* The results characterise the generator, not real fraud; the README, model card and
  governance doc say so prominently.
* Integration with the pipeline is conceptual and documented in `docs/DATA_CONTRACT.md`.
* Anyone can reproduce every number in the docs with one command.
