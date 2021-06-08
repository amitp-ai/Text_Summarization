# The "buster" flavor of the official docker Python image is based on Debian and includes common packages.
FROM python:3.6-buster

# Print commands and their arguments as they are executed and Create the working directory
RUN set -ex && mkdir /mlops
WORKDIR /mlops

# Install Python dependencies
COPY requirements.txt ./requirements.txt
# RUN sed -i 's/cu101/cpu/' requirements.txt
RUN pip install --upgrade pip~=21.0.0
RUN pip install -r requirements.txt

# Copy only the relevant directories and files
COPY src/ ./src
COPY Data/*.csv ./Data/
COPY Data/*.json ./Data/
copy Data/Training/Dicts ./Data/Training/Dicts
COPY logs/ ./logs
COPY config.yaml ./config.yaml
COPY s3Bucket.sh ./s3Bucket.sh

# Expose port
EXPOSE 8000
ENV PYTHONPATH /mlops
CMD python3 # /mlops/src/inference.py

