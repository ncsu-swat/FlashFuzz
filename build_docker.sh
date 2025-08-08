#!/bin/bash

# tf 2.19
docker build -t ncsuswat/flashfuzz:tf2.19-base -f docker/tf2.19-base.Dockerfile .

# tf 2.16
docker build -t ncsuswat/flashfuzz:tf2.16-base -f docker/tf2.16-base.Dockerfile .

# tf 2.16-fuzz
docker build -t ncsuswat/flashfuzz:tf2.16-fuzz -f docker/tf2.16-fuzz.Dockerfile .

# tf 2.16-cov
docker build -t ncsuswat/flashfuzz:tf2.16-cov -f docker/tf2.16-cov.Dockerfile .
