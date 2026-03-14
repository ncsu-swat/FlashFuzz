FROM ncsuswat/flashfuzz:tf2.16-cov

WORKDIR /root/tensorflow/fuzz

# Remove existing harnesses and replace with GLM-5 generated ones
RUN rm -rf tf.*

COPY testharness/tf_cpu_glm5 /root/tensorflow/fuzz
COPY scripts/ .

RUN python3 -u build_test_harness.py --dll tf --mode cov --no-compile

WORKDIR /root

CMD ["bash"]
