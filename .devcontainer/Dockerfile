# syntax=docker/dockerfile:1

# Use the specified base image
FROM mcr.microsoft.com/vscode/devcontainers/python:3.11

# Set the working directory inside the container
WORKDIR /app

# Install any global dependencies or tools you might need for development
RUN apt-get update && apt-get install libleveldb-dev -y