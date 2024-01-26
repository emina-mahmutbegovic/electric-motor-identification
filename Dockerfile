# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited

# Use an official Python runtime as a parent image
FROM python:3.11.7

LABEL authors="emina.mahmutbegovic"

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install required system dependencies
RUN apt-get update && apt-get install -y qtbase5-dev \
    qtchooser \
    qt5-qmake \
    qtbase5-dev-tools \
    && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Run main.py when the container launches
CMD ["python3", "./main.py"]


