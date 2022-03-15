# FROM maxxiking/fc_19_3_gmsh_py_39_env:1.0.0
FROM maxxiking/fc_19_3_of_9:latest

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

ENV PYTHONPATH="${PYTHONPATH}:/tmp/squashfs-root/usr/lib/python3.9/site-packages/:/tmp/squashfs-root/usr/lib/"
