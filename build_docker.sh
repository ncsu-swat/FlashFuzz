#!/bin/bash

MEM_LIMIT="128g"
# tf 2.19
docker build -m ${MEM_LIMIT}  -t ncsuswat/flashfuzz:tf2.19-base -f docker/tf2.19-base.Dockerfile .

# tf 2.13
docker build -m ${MEM_LIMIT}  -t ncsuswat/flashfuzz:tf2.16-base -f docker/tf2.16-base.Dockerfile .

# tf 2.13-fuzz
docker build -m ${MEM_LIMIT}  -t ncsuswat/flashfuzz:tf2.16-fuzz -f docker/tf2.16-fuzz.Dockerfile .

# tf 2.13-cov
docker build -m ${MEM_LIMIT}  -t ncsuswat/flashfuzz:tf2.16-cov -f docker/tf2.16-cov.Dockerfile .
