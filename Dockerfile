# Use a lightweight Python base image
FROM python:3.12-slim 

# Set working directory
WORKDIR /app

# Create a non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the necessary application files
COPY main.py .
COPY config.ini .
COPY utils/ utils/

# Change ownership of the application directory to the non-root user
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Entrypoint: adjust if you use a specific runner
CMD ["python", "main.py"]
