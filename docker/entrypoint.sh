#!/bin/sh
# Container entrypoint. If no trusted model artifact is present and bootstrapping is enabled,
# train one from synthetic data first so that the API becomes ready without any external input.
# Commands that train anyway (demo, train) skip the bootstrap.
set -e

MODEL_DIR="${FRAUD_ML_MODEL_DIR:-/app/models}"
case "${2:-}" in
  demo|train|generate-data|version) NEEDS_MODEL=false ;;
  *) NEEDS_MODEL=true ;;
esac

if [ "$NEEDS_MODEL" = "true" ] && [ "${FRAUD_ML_BOOTSTRAP_MODEL:-true}" = "true" ] \
   && [ ! -f "${MODEL_DIR}/latest/metadata.json" ]; then
  echo "no model artifact in ${MODEL_DIR}/latest; training a bootstrap model from synthetic data" >&2
  fraud-ml --log-format json demo --workdir /app --fast --no-mlflow \
    --transactions "${FRAUD_ML_BOOTSTRAP_ROWS:-20000}" --seed "${FRAUD_ML_BOOTSTRAP_SEED:-42}" >&2
fi

exec "$@"
