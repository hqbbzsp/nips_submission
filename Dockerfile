# Use an official Python runtime as a parent image
# other options in https://github.com/orgs/pytorch/packages/container/pytorch-nightly/versions?filters%5Bversion_type%5D=tagged
# Lit-GPT requires current nightly (future 2.1) for the latest attention changes
FROM python:3.8.0

# Set the working directory in the container to /submission
WORKDIR /submission

RUN pip install --upgrade pip 

RUN pip3 install torch torchvision torchaudio 
# Copy the specific file into the container at /submission
COPY ./final_model/9.13_100steps/ /submission/final_model/9.13_100steps/

# Setup server requriements
COPY ./fast_api_requirements.txt fast_api_requirements.txt
RUN pip3 install --no-cache-dir --upgrade -r fast_api_requirements.txt

COPY ./requirements.txt requirements.txt
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

# Install any needed packages specified in requirements.txt that come from lit-gpt
RUN apt-get update && apt-get install -y git
RUN pip3 install huggingface_hub sentencepiece trl lightning


# Copy over single file server
COPY ./main_refer.py /submission/main_refer.py
COPY ./helper1.py /submission/helper1.py
COPY ./api.py /submission/api.py
# Run the server
CMD ["uvicorn", "main_refer:app", "--host", "0.0.0.0", "--port", "80"]