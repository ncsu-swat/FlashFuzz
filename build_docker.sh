#!/bin/bash

# tf 2.19
docker build -t ncsuswat/flashfuzz:tf2.19-base -f docker/tf2.19-base.Dockerfile .

# tf 2.13
docker build -t ncsuswat/flashfuzz:tf2.13-base -f docker/tf2.13-base.Dockerfile .

# tf 2.13-cov
docker build -t ncsuswat/flashfuzz:tf2.13-cov -f docker/tf2.13-cov.Dockerfile .
