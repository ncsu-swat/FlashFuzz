FROM ncsuswat/flashfuzz:torch2.2-fuzz

WORKDIR /root/fuzz

# Remove existing harnesses and replace with GLM-5 generated ones
RUN rm -rf torch.*

COPY scripts /root/fuzz/
COPY testharness/torch_cpu_glm5 /root/fuzz

RUN python3 -u build_test_harness.py --dll torch --mode fuzz

WORKDIR /root

CMD ["bash"]
