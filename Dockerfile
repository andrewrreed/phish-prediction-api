# build anaconda environment 
FROM continuumio/miniconda3:latest
COPY ./environment.yaml /env/
RUN conda config --add channels anaconda
RUN conda config --append channels conda-forge
RUN conda config --append channels defaults
RUN conda env create -f /env/environment.yaml

# activate conda and set path
RUN echo "source activate phish-api" > ~/.bashrc
ENV PATH /opt/conda/envs/phish-api/bin:$PATH

# copy in api app files and models
COPY ./api.py /deploy/
COPY ./util.py /deploy/
COPY ./models/* /models/

# set working dir and expose port 5000
EXPOSE 5000
ENTRYPOINT ["python", "deploy/api.py"]