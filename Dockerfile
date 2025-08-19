# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install Rust for compiling dependencies like 'tokenizers'
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
# Add Rust to the PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port Hugging Face Spaces uses
EXPOSE 7860

# Define the command to run your app on the correct port
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:7860", "--timeout", "120"]
