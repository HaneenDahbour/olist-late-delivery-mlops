FROM python:3.12-slim

WORKDIR /workspace

RUN pip install --no-cache-dir "dvc[s3]==3.67.1"

ENTRYPOINT ["/bin/sh", "-c"]
# Best-effort: if the remote has the objects, fetch them. If the remote is
# empty or unreachable (e.g. a zipped/offline copy of this repo, shipped
# with the real data/artifacts files already in place instead of via DVC),
# don't block the rest of the stack — register-model will fail with a
# clear FileNotFoundError if the files genuinely aren't present either way.
CMD ["dvc pull || echo 'dvc pull found nothing to fetch (remote empty/unreachable) — continuing with whatever is already on disk'"]
