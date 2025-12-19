FROM nvcr.io/nvidia/rapidsai/base:23.12-cuda12.0-py3.10

# Set the working directory
WORKDIR /app

# Copy only the pyproject.toml file
COPY ./pyproject.toml /app/

RUN pip install .