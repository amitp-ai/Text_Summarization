# The "buster" flavor of the official docker Python image is based on Debian and includes common packages.
FROM python:3.6-buster

# Print commands and their arguments as they are executed and Create the working directory
RUN set -ex && mkdir /production
WORKDIR /production

# Install Python dependencies
COPY requirements.txt ./requirements.txt
# RUN sed -i 's/cu101/cpu/' requirements.txt
RUN pip install --upgrade pip~=21.0.0
RUN pip install -r requirements.txt

# Copy only the relevant directories and files (using .dockerignore)
COPY ./ ./

# Expose port and command to run
EXPOSE 8000
ENV PYTHONPATH /production
CMD python3 # /production/src/inference.py

