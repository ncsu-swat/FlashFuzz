#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/nn_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/framework/graph.pb.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        // Need at least basic parameters
        if (size < 50) return 0;
        
        // Extract dimensions for input tensor [batch, depth, rows, cols, channels]
        int batch = (data[offset] % 4) + 1;
        offset++;
        int depth = (data[offset] % 8) + 1;
        offset++;
        int rows = (data[offset] % 16) + 1;
        offset++;
        int cols = (data[offset] % 16) + 1;
        offset++;
        int channels = (data[offset] % 8) + 1;
        offset++;
        
        // Extract ksize parameters (must have ksize[0] = ksize[4] = 1)
        int ksize_depth = (data[offset] % 3) + 1;
        offset++;
        int ksize_rows = (data[offset] % 3) + 1;
        offset++;
        int ksize_cols = (data[offset] % 3) + 1;
        offset++;
        
        // Extract strides parameters (must have strides[0] = strides[4] = 1)
        int stride_depth = (data[offset] % 3) + 1;
        offset++;
        int stride_rows = (data[offset] % 3) + 1;
        offset++;
        int stride_cols = (data[offset] % 3) + 1;
        offset++;
        
        // Extract padding type
        bool use_same_padding = (data[offset] % 2) == 0;
        offset++;
        
        // Extract data format
        bool use_ndhwc = (data[offset] % 2) == 0;
        offset++;
        
        // Extract data type
        tensorflow::DataType dtype;
        int dtype_choice = data[offset] % 3;
        offset++;
        switch (dtype_choice) {
            case 0: dtype = tensorflow::DT_HALF; break;
            case 1: dtype = tensorflow::DT_BFLOAT16; break;
            default: dtype = tensorflow::DT_FLOAT; break;
        }
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input tensor shape
        tensorflow::TensorShape input_shape;
        if (use_ndhwc) {
            input_shape = tensorflow::TensorShape({batch, depth, rows, cols, channels});
        } else {
            input_shape = tensorflow::TensorShape({batch, channels, depth, rows, cols});
        }
        
        // Create input tensor
        tensorflow::Tensor input_tensor(dtype, input_shape);
        
        // Fill tensor with data from fuzzer input
        if (dtype == tensorflow::DT_FLOAT) {
            auto flat = input_tensor.flat<float>();
            for (int i = 0; i < flat.size() && offset < size; i++, offset++) {
                flat(i) = static_cast<float>(data[offset]) / 255.0f;
            }
        } else if (dtype == tensorflow::DT_HALF) {
            auto flat = input_tensor.flat<Eigen::half>();
            for (int i = 0; i < flat.size() && offset < size; i++, offset++) {
                flat(i) = Eigen::half(static_cast<float>(data[offset]) / 255.0f);
            }
        } else { // DT_BFLOAT16
            auto flat = input_tensor.flat<tensorflow::bfloat16>();
            for (int i = 0; i < flat.size() && offset < size; i++, offset++) {
                flat(i) = tensorflow::bfloat16(static_cast<float>(data[offset]) / 255.0f);
            }
        }
        
        // Create input placeholder
        auto input_ph = tensorflow::ops::Placeholder(root, dtype);
        
        // Set up ksize and strides (must have first and last elements as 1)
        std::vector<int> ksize = {1, ksize_depth, ksize_rows, ksize_cols, 1};
        std::vector<int> strides = {1, stride_depth, stride_rows, stride_cols, 1};
        
        // Set padding string
        std::string padding = use_same_padding ? "SAME" : "VALID";
        
        // Set data format string
        std::string data_format = use_ndhwc ? "NDHWC" : "NCDHW";
        
        // Create MaxPool3D operation
        auto maxpool3d = tensorflow::ops::MaxPool3D(
            root, 
            input_ph, 
            ksize, 
            strides, 
            padding,
            tensorflow::ops::MaxPool3D::DataFormat(data_format)
        );
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run(
            {{input_ph, input_tensor}}, 
            {maxpool3d}, 
            &outputs
        );
        
        // Check if operation succeeded
        if (!status.ok()) {
            std::cout << "MaxPool3D operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output tensor properties
        if (!outputs.empty()) {
            const tensorflow::Tensor& output = outputs[0];
            if (output.dtype() != dtype) {
                std::cout << "Output dtype mismatch" << std::endl;
                return 0;
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}