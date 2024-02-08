FROM ubuntu:20.04

RUN apt-get update
RUN apt-get install python3-pip -y

# Create app directory
WORKDIR /app

ADD ./DATA/src/ ./src
ADD ./DATA/data ./raw_data

COPY ./DATA/requirements.txt ./src/requirements.txt

# Install the Python dependencies
RUN pip3 install --no-cache-dir --upgrade -r ./src/requirements.txt

RUN mkdir log_file
RUN mkdir data
RUN mkdir model

EXPOSE 8000

CMD ["python3", "src/main_data.py"]