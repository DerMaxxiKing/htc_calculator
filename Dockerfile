# FROM maxxiking/fc_19_3_gmsh_py_39_env:1.0.0
FROM maxxiking/fc_of9_py39:latest

COPY requirements.txt /tmp/requirements.txt
RUN python3.9 - m pip3 install -r /tmp/requirements.txt
