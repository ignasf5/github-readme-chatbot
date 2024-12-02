# Use an official Python runtime as a parent image
FROM python:3.13.0

# Set the working directory in the container
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app/

# Expose port 8501 for the Streamlit app
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app.py"]
