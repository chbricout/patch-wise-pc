FROM nvcr.io/nvidia/rapidsai/base:23.12-cuda12.0-py3.11

# Set the working directory
WORKDIR /app

RUN python -m pip install --upgrade pip

COPY pyproject.toml ./

RUN pip install --no-cache-dir .

