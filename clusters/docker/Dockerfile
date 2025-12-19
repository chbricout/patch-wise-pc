FROM nvcr.io/nvidia/rapidsai/base:25.12a-cuda12-py3.11

# Set the working directory
WORKDIR /app

RUN python -m pip install --upgrade pip

COPY pyproject.toml ./

RUN pip install --no-cache-dir .

