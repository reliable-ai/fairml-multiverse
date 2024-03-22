FROM python:3.8.18-slim-bookworm

# Install dependencies for python packages
RUN apt-get update && \
    apt-get install -y gcc git  && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in Pipfile
RUN pip install pipenv
RUN pipenv sync -d

# Define environment variable
ENV SEED 2023

# Run multiverse_analysis.py when the container launches, passing the seed as an argument
CMD ["python", "multiverse_analysis.py", "--seed", "$SEED"]
