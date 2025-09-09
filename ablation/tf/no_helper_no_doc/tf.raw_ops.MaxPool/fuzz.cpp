#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/kernel_def_builder.h>
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract parameters from fuzzer input
        int batch_size = (data[offset] % 4) + 1;
        offset++;
        int height = (data[offset] % 32) + 1;
        offset++;
        int width = (data[offset] % 32) + 1;
        offset++;
        int channels = (data[offset] % 8) + 1;
        offset++;
        
        int ksize_h = (data[offset] % 8) + 1;
        offset++;
        int ksize_w = (data[offset] % 8) + 1;
        offset++;
        
        int stride_h = (data[offset] % 4) + 1;
        offset++;
        int stride_w = (data[offset] % 4) + 1;
        offset++;
        
        int padding_type = data[offset] % 2;
        offset++;
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input tensor shape
        tensorflow::TensorShape input_shape({batch_size, height, width, channels});
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, input_shape);
        
        // Fill input tensor with fuzzer data
        auto input_flat = input_tensor.flat<float>();
        for (int i = 0; i < input_flat.size() && offset < size; i++) {
            if (offset + 3 < size) {
                float val;
                memcpy(&val, &data[offset], sizeof(float));
                input_flat(i) = val;
                offset += 4;
            } else {
                input_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
                offset++;
            }
        }
        
        // Create input placeholder
        auto input_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        // Set up MaxPool parameters
        std::vector<int> ksize = {1, ksize_h, ksize_w, 1};
        std::vector<int> strides = {1, stride_h, stride_w, 1};
        std::string padding = (padding_type == 0) ? "VALID" : "SAME";
        
        // Create MaxPool operation
        auto maxpool = tensorflow::ops::MaxPool(root, input_ph, ksize, strides, padding);
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({{input_ph, input_tensor}}, {maxpool}, &outputs);
        
        if (!status.ok()) {
            std::cout << "MaxPool operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output tensor is valid
        if (!outputs.empty()) {
            const tensorflow::Tensor& output = outputs[0];
            if (output.NumElements() > 0) {
                // Basic validation - check if output has reasonable dimensions
                auto output_shape = output.shape();
                if (output_shape.dims() == 4 && 
                    output_shape.dim_size(0) == batch_size &&
                    output_shape.dim_size(3) == channels) {
                    // Success case
                }
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}