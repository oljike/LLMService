FROM nvidia/cuda:11.1-base
RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . /app
WORKDIR /app
CMD ["python3", "run_llm.py"]
