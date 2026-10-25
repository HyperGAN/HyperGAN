# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

COPY pyproject.toml setup.py MANIFEST.in README.md LICENSE ./
COPY src/ ./src/

# .dockerignore keeps .git out of the build context, so the wheel cannot read
# its commit. Pass it in: --build-arg HYPERGAN_COMMIT="$(git rev-parse HEAD)".
# Without it the image records the commit as unknown (null).
ARG HYPERGAN_COMMIT=""

RUN set -eu; \
    if [ -n "${HYPERGAN_COMMIT}" ]; then \
        printf '{"commit": "%s", "dirty": null, "schema_version": 1}\n' "${HYPERGAN_COMMIT}" \
            > src/hypergan/_build_provenance.json; \
    fi; \
    python -m pip wheel --no-cache-dir --no-deps --wheel-dir /dist .


FROM python:3.12-slim-bookworm AS runtime

ARG EXTRAS=""

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HYPERGAN_STATE_DIR=/runs/.hypergan

RUN groupadd --gid 1000 hypergan \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash hypergan \
    && mkdir -p /data /runs /workspace \
    && chown -R hypergan:hypergan /data /runs /workspace

COPY --from=builder /dist/*.whl /tmp/dist/
RUN set -eux; \
    wheel="$(ls /tmp/dist/*.whl)"; \
    if [ -n "${EXTRAS}" ]; then \
        python -m pip install --no-cache-dir "${wheel}[${EXTRAS}]"; \
    else \
        python -m pip install --no-cache-dir "${wheel}"; \
    fi; \
    rm -rf /tmp/dist

USER hypergan:hypergan
WORKDIR /workspace
VOLUME ["/data", "/runs"]
EXPOSE 8765
STOPSIGNAL SIGINT

ENTRYPOINT ["python", "-m", "hypergan"]
CMD ["--help"]
