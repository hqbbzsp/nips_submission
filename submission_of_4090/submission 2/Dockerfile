FROM python:3.8.0

RUN apt-get update  && apt-get install -y git python3-virtualenv wget

WORKDIR /submission 

RUN pip3 install torch torchvision torchaudio 

COPY ./fast_api_requirements.txt fast_api_requirements.txt
RUN pip3 install --no-cache-dir --upgrade -r fast_api_requirements.txt 

COPY ./require.txt require.txt
RUN pip3 install --no-cache-dir --upgrade -r require.txt 

ENV HUGGINGFACE_TOKEN="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"

# Copy over single file server
COPY ./main.py main.py
COPY ./api.py api.py
COPY ./merge.py merge.py
# Run the server
CMD ["/bin/bash", "-e", "-c", "python merge.py && uvicorn main:app --host 0.0.0.0 --port 80"]