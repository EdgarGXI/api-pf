# Use ultralytics base image
FROM ultralytics/ultralytics:latest

# Set the working directory
WORKDIR /usr/src/app

# Install dependencies
COPY ./app/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app/best.pt ./best.pt
COPY ./app/best2.pt ./best2.pt
COPY ./app/best3.pt ./best3.pt
# Copy the rest of the code
COPY ./app /usr/src/app/app

# Expose the port the app runs on
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]