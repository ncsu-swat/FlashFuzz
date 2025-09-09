#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/platform/env.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 20) return 0;
        
        // Extract dimensions from fuzzer input
        uint8_t batch_size = (data[offset++] % 3) + 1;
        uint8_t rows = (data[offset++] % 5) + 2;
        uint8_t cols = (data[offset++] % 5) + 2;
        
        // Extract k values
        int8_t k_low = (int8_t)(data[offset++] % 5) - 2;
        int8_t k_high = k_low + (data[offset++] % 3);
        
        // Extract alignment
        uint8_t align_idx = data[offset++] % 4;
        std::string align;
        switch (align_idx) {
            case 0: align = "RIGHT_LEFT"; break;
            case 1: align = "LEFT_RIGHT"; break;
            case 2: align = "LEFT_LEFT"; break;
            case 3: align = "RIGHT_RIGHT"; break;
        }
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input tensor
        tensorflow::TensorShape input_shape({batch_size, rows, cols});
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, input_shape);
        auto input_flat = input_tensor.flat<float>();
        
        // Fill input tensor with fuzzer data
        for (int i = 0; i < input_flat.size() && offset < size; ++i) {
            input_flat(i) = static_cast<float>(data[offset++] % 100) / 10.0f;
        }
        
        // Calculate diagonal dimensions
        int num_diags = k_high - k_low + 1;
        int max_diag_len = std::min(rows + std::min(-k_low, 0), cols + std::min(k_high, 0));
        
        tensorflow::TensorShape diag_shape;
        if (k_low == k_high) {
            diag_shape = tensorflow::TensorShape({batch_size, max_diag_len});
        } else {
            diag_shape = tensorflow::TensorShape({batch_size, num_diags, max_diag_len});
        }
        
        // Create diagonal tensor
        tensorflow::Tensor diagonal_tensor(tensorflow::DT_FLOAT, diag_shape);
        auto diag_flat = diagonal_tensor.flat<float>();
        
        // Fill diagonal tensor with fuzzer data
        for (int i = 0; i < diag_flat.size() && offset < size; ++i) {
            diag_flat(i) = static_cast<float>(data[offset++] % 100) / 10.0f;
        }
        
        // Create k tensor
        tensorflow::TensorShape k_shape;
        tensorflow::Tensor k_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        if (k_low == k_high) {
            k_tensor.scalar<int32_t>()() = k_low;
        } else {
            k_tensor = tensorflow::Tensor(tensorflow::DT_INT32, tensorflow::TensorShape({2}));
            auto k_flat = k_tensor.flat<int32_t>();
            k_flat(0) = k_low;
            k_flat(1) = k_high;
        }
        
        // Create placeholders
        auto input_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto diagonal_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto k_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        
        // Create MatrixSetDiagV3 operation
        auto matrix_set_diag = tensorflow::ops::MatrixSetDiagV3(
            root, input_ph, diagonal_ph, k_ph,
            tensorflow::ops::MatrixSetDiagV3::Align(align)
        );
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run(
            {{input_ph, input_tensor}, {diagonal_ph, diagonal_tensor}, {k_ph, k_tensor}},
            {matrix_set_diag},
            &outputs
        );
        
        if (!status.ok()) {
            std::cout << "TensorFlow operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output shape
        if (!outputs.empty()) {
            const auto& output = outputs[0];
            if (output.shape().dims() != 3 ||
                output.shape().dim_size(0) != batch_size ||
                output.shape().dim_size(1) != rows ||
                output.shape().dim_size(2) != cols) {
                std::cout << "Unexpected output shape" << std::endl;
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