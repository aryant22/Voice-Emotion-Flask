# Use the python:3.12-slim base image
FROM python:3.12-slim

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    llvm \
    llvm-dev \
    musl-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Check if llvm-config exists, if not create a symbolic link
RUN if [ ! -f /usr/bin/llvm-config ]; then ln -s $(which llvm-config) /usr/bin/llvm-config; fi

# Upgrade pip
RUN pip install --upgrade pip

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file used for dependencies
COPY requirements.txt .

# Install numpy separately
RUN pip install numpy

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the rest of the working directory contents into the container at /app
COPY . .

# Run app.py when the container launches
ENTRYPOINT ["python", "app.py"]
