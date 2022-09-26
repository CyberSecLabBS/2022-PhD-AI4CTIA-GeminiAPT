# syntax=docker/dockerfile:1

FROM python:3.7

WORKDIR /app

COPY ./requirements.txt ./requirements.txt
# COPY ./ASTERIA/ASTExtraction/requirements.txt ./requirements2.txt
# COPY /Users/gianlucacapozzi/Documents/RICERCA/data/fakeDB/DB/ .

ADD ./DB.tar.xz .

RUN pip3 install -r requirements.txt
# RUN pip3 install -r requirements2.txt

# INSTALL RADARE2
RUN git clone https://github.com/radareorg/radare2
RUN radare2/sys/install.sh

# INSTALL SQLITE (FOR ASTERIA)
# RUN apt-get -y update
# RUN apt-get -y upgrade
# RUN apt-get install -y sqlite3 libsqlite3-dev