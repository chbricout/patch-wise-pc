FROM rapidsai/base:25.12-cuda12-py3.11
USER root
# Set the working directory
WORKDIR /app

RUN apt-get update && apt-get install build-essential -y
RUN python -m pip install --upgrade pip

COPY pyproject.toml ./

RUN pip install --no-cache-dir .

