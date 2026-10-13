# Security policy

This repository is an independent educational project that processes **synthetic data
only**. It is not deployed anywhere and must not be used for production fraud decisioning.

* Please report any vulnerability in the code, the container image or the CI configuration
  by opening a GitHub issue titled `security:` with a description of the problem. Do not
  include real customer or payment data in reports; none is used or needed here.
* Known limitations by design: the API has no authentication, model artifacts are loaded
  from the local filesystem only, and all identifiers are synthetic.

Details of the trust boundaries, the pickle/joblib handling and the API controls are in
[docs/SECURITY.md](docs/SECURITY.md).
