FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      python3.11 python3.11-venv python3.11-dev python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    rm -rf /var/lib/apt/lists/*


# Set the working directory
WORKDIR /app

RUN python -m pip install --upgrade pip

COPY pyproject.toml ./

RUN pip install --no-cache-dir .

