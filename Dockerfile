# Edited from https://stackoverflow.com/a/64524896
FROM r-base:4.0.5
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        gdb lcov pkg-config \
        libbz2-dev libffi-dev libgdbm-dev libgdbm-compat-dev liblzma-dev \
        libncurses5-dev libreadline6-dev libsqlite3-dev libssl-dev \
        lzma lzma-dev tk-dev uuid-dev zlib1g-dev \
        git \
        vim \
 && rm -rf /var/lib/apt/lists/*
RUN wget -q https://www.python.org/ftp/python/3.9.10/Python-3.9.10.tgz \
 && tar -xf Python-3.9.10.tgz \
 && cd Python-3.9.10 \
 && ./configure --enable-optimizations \
 && grep ssl Modules/Setup | sed 's/^#//' >> Modules/Setup.local \
 && make -j 2 \
 && make altinstall \
 && cd .. \
 && rm -r Python-3.9.10.tgz Python-3.9.10
RUN ln -s /usr/local/bin/python3.9 /usr/local/bin/python \
 && ln -s /usr/local/bin/pip3.9 /usr/local/bin/pip3
RUN pip3 install --upgrade pip 

WORKDIR /app

# installing python libraries
RUN pip3 install --no-cache-dir --prefer-binary rpy2
COPY build/requirements.txt .
RUN pip3 install --no-cache-dir --prefer-binary -r requirements.txt

# installing R libraries
COPY build/requirements.R .
RUN Rscript requirements.R
