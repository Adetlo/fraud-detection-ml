# Security policy

## Scope

`fraud-detection-ml` is an independent **educational project**. It processes **synthetic
financial data only**: no real customer, account, card or payment data is used, needed or
accepted, and it must not be used for production fraud decisioning, regulatory compliance or
autonomous blocking of transactions. It is not deployed anywhere and carries no production
financial workload. The threat model is that of an open-source developer tool: supply-chain
issues in dependencies, unsafe handling of local model artifacts, and code defects that could
mislead someone who adapts the project for real data.

Known limitations by design: the API has no authentication (OAuth2/OIDC would be required for
any real deployment), model artifacts are loaded from the local filesystem only after a hash
check, and all identifiers are synthetic. The trust boundaries, the joblib/pickle handling
and the API controls are described in [docs/SECURITY.md](docs/SECURITY.md).

## Supported versions

Only the `main` branch receives fixes. There are no releases or backports.

## Reporting a vulnerability

1. Please do **not** open a public issue with exploit details.
2. Use GitHub's private vulnerability reporting on the repository
   (**Security** tab, **Report a vulnerability**) so the report stays confidential.
3. Include the affected component (code, container image or CI configuration), a minimal
   reproduction and the impact you believe it has. Do not include real customer or payment
   data in reports; none is used here.

This is a volunteer-maintained portfolio project: expect a best-effort acknowledgement within
a couple of weeks, and please allow time for a fix before public disclosure. There is no bug
bounty and no e-mail contact for this project.

## In scope

* Code under `src/`, `scripts/`, the `Dockerfile` and the CI workflows.
* Dependency vulnerabilities surfaced by `pip-audit` that affect the service or the training path.

## Out of scope

* Findings that require real financial data, real credentials or production infrastructure.
* Missing authentication on the API (documented as out of scope for v1).
* Hardening a production deployment would need (encryption, access control, secret management,
  model signing); these are documented as not implemented.
* Denial-of-service against a locally run demo stack.
