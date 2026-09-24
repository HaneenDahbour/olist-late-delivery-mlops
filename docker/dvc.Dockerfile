FROM python:3.12-slim

WORKDIR /workspace

RUN pip install --no-cache-dir "dvc[s3]==3.67.1"

ENTRYPOINT ["dvc"]
CMD ["pull"]
