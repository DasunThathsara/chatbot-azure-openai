FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main application file
COPY main.py .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable to turn off python output buffering
ENV PYTHONUNBUFFERED 1

# Run main.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
