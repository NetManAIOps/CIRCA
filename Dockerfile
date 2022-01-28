# Edited from https://stackoverflow.com/a/64524896
FROM r-base:4.0.5
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        vim \
        python3.9 \
        python3-pip \
        python3-setuptools \
        python3-dev \
 && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip 

WORKDIR /app

# installing python libraries
RUN pip3 install --no-cache-dir rpy2
COPY build/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# installing R libraries
COPY build/requirements.R .
RUN Rscript requirements.R
