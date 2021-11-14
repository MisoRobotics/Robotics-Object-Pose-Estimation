FROM  nvcr.io/nvidia/pytorch:21.10-py3

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
    apt-get clean

# Add Tini init systems to handle orphaned processes
# https://cloud.google.com/solutions/best-practices-for-building-containers#problem_2_how_classic_init_systems_handle_orphaned_processes
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/local/bin/tini
RUN chmod +x /usr/local/bin/tini

# Add project stuff
WORKDIR /
VOLUME /notebooks

# This will install all dependencies
COPY environment-gpu.yml ./
RUN conda env update -n base -f environment-gpu.yml && \
    conda clean -afy

# COPY the rest of code here
COPY . ./

# pip install this package
RUN pip install -e .

# pre-load VGG weights
RUN python -c 'import pose_estimation.model as model; model.preload()'

# Use -g to ensure all child process received SIGKILL
ENTRYPOINT ["tini", "-g", "--"]

CMD sh -c "jupyter notebook --notebook-dir=/ --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX}"
