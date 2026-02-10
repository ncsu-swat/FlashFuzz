#!/bin/bash

# # tf 2.19
docker build -t ncsuswat/flashfuzz:tf2.19-base -f docker/tf2.19-base.Dockerfile .

# tf 2.19-fuzz
docker build -t ncsuswat/flashfuzz:tf2.19-fuzz -f docker/tf2.19-fuzz.Dockerfile .

# # tf 2.16
docker build -t ncsuswat/flashfuzz:tf2.16-base -f docker/tf2.16-base.Dockerfile .

# # tf 2.16-fuzz
docker build -t ncsuswat/flashfuzz:tf2.16-fuzz -f docker/tf2.16-fuzz.Dockerfile .

# # tf 2.16-cov
docker build -t ncsuswat/flashfuzz:tf2.16-cov -f docker/tf2.16-cov.Dockerfile .

# torch 2.2
docker build -t ncsuswat/flashfuzz:torch2.2-base -f docker/torch-2.2-base.Dockerfile .

# torch 2.2-fuzz
docker build -t ncsuswat/flashfuzz:torch2.2-fuzz -f docker/torch-2.2-fuzz.Dockerfile .

# torch 2.2-cov
docker build -t ncsuswat/flashfuzz:torch2.2-cov -f docker/torch-2.2-cov.Dockerfile .

# torch 2.7
docker build -t ncsuswat/flashfuzz:torch2.7-base -f docker/torch-2.7-base.Dockerfile .

# torch 2.7-fuzz
docker build -t ncsuswat/flashfuzz:torch2.7-fuzz -f docker/torch-2.7-fuzz.Dockerfile .

# torch 2.10
docker build -t ncsuswat/flashfuzz:torch2.10-base -f docker/torch-2.10-base.Dockerfile .

# torch 2.10-fuzz
docker build -t ncsuswat/flashfuzz:torch2.10-fuzz -f docker/torch-2.10-fuzz.Dockerfile .
