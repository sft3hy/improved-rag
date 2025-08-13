FROM python:3.13.5-alpine

# Update packages and install necessary dependencies
RUN apk update && \
    apk add --no-cache wget && \
    apk add --no-cache build-base libffi-dev openssl-dev && \
    apk add python3 py3-pip gcc musl-dev


# Install Python packages from requirements.txt (if needed)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# Set the working directory
WORKDIR /app


# Copy your application code into the container
COPY . /app


# Specify the command to run your Python application
ENV PYTHONUNBUFFERED=1
# or run with -u
CMD ["python", "-u", "/app/app.py"]
