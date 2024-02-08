FROM ubuntu:20.04


RUN apt-get update
RUN apt-get install python3-pip -y

# Create app directory
WORKDIR /app

ADD src/ ./src

COPY ./requirements.txt ./src/requirements.txt

# Install the Python dependencies
RUN pip3 install --no-cache-dir --upgrade -r ./src/requirements.txt

RUN mkdir log_file
RUN mkdir data
RUN mkdir model

CMD ["python3", "src/main.py"]