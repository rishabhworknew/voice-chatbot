# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependency file and install dependencies
# This is done first to leverage Docker's layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Make port 8765 available to the world outside this container
EXPOSE 8765

# Define the command to run your application
# The server is bound to 0.0.0.0 to be accessible from outside the container
CMD ["python", "main.py"]