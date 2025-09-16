# Use an official lightweight Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements.txt first for dependency installation
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files
COPY . .

# Expose port (change if your app uses a different port)
EXPOSE 5000

# Command to run your app (modify as needed)
CMD ["python", "app.py"]