# Use an official Python runtime as a parent image
FROM python:3.10.18-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip to ensure the latest version
RUN pip install --upgrade pip

# Install any system dependencies required by Python packages
# This is crucial for 'slim' images if packages in requirements.txt
# have underlying C/C++ dependencies or need build tools.
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install your Python packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Streamlit runs on (default is 8501)
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "app.py"]