clang++ -fsanitize=fuzzer \
         -fno-omit-frame-pointer \
         -O0 -g \
         -I/root/pytorch/build-fuzz/include \
         -I/root/pytorch/aten/src \
         -I/root/pytorch/c10/core \
         -I/root/pytorch \
         -I/root/pytorch/build-fuzz \
         -I/root/pytorch/build-fuzz/aten/src \
         -I/root/pytorch/torch/csrc/api/include \
         -I/usr/local/cuda/include \
         -std=c++17 \
         -I/. \
         main.cpp fuzzer_utils.cpp \
         -Wl,-rpath,/root/pytorch/build-fuzz/caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen \
         -L/root/pytorch/build-fuzz/caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen \
         -Wl,-rpath,/root/pytorch/build-fuzz/c10/CMakeFiles/c10.dir/core/ \
         -L/root/pytorch/build-fuzz/c10/CMakeFiles/c10.dir/core/ \
         -Wl,-rpath,/root/pytorch/build-fuzz/lib \
         -L/root/pytorch/build-fuzz/lib \
         -Wl,-rpath,/root/pytorch/build/lib \
         -L/root/pytorch/build/lib \
         -Wl,-rpath,/root/pytorch/build/lib \
         -ltorch -ltorch_cpu  \
          -lc10 \
         -o fuzz

if [ $? -ne 0 ]; then
    echo "Error: Compilation failed!"
    exit 1
fi

python3 random_seed.py
