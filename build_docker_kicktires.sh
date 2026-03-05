#!/bin/bash
set -ex

# Build only the PyTorch 2.2 images needed for the kick-the-tires evaluation
docker build -t ncsuswat/flashfuzz:torch2.2-base -f docker/torch-2.2-base.Dockerfile .
docker build -t ncsuswat/flashfuzz:torch2.2-fuzz -f docker/torch-2.2-fuzz.Dockerfile .
