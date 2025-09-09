#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/sparse_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract dimensions and parameters
        int32_t nnz = (data[offset] % 10) + 1;
        offset++;
        int32_t rows = (data[offset] % 10) + 1;
        offset++;
        int32_t cols = (data[offset] % 10) + 1;
        offset++;
        int32_t b_cols = (data[offset] % 10) + 1;
        offset++;
        bool adjoint_a = data[offset] % 2;
        offset++;
        bool adjoint_b = data[offset] % 2;
        offset++;
        
        if (offset + nnz * 2 * sizeof(int32_t) + nnz * sizeof(float) + rows * cols * sizeof(float) > size) {
            return 0;
        }
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create a_indices tensor [nnz, 2]
        tensorflow::Tensor a_indices_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({nnz, 2}));
        auto a_indices_flat = a_indices_tensor.flat<int32_t>();
        for (int i = 0; i < nnz; i++) {
            a_indices_flat(i * 2) = (reinterpret_cast<const int32_t*>(data + offset)[i * 2]) % rows;
            a_indices_flat(i * 2 + 1) = (reinterpret_cast<const int32_t*>(data + offset)[i * 2 + 1]) % cols;
        }
        offset += nnz * 2 * sizeof(int32_t);
        
        // Create a_values tensor [nnz]
        tensorflow::Tensor a_values_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({nnz}));
        auto a_values_flat = a_values_tensor.flat<float>();
        for (int i = 0; i < nnz; i++) {
            a_values_flat(i) = reinterpret_cast<const float*>(data + offset)[i];
        }
        offset += nnz * sizeof(float);
        
        // Create a_shape tensor [2]
        tensorflow::Tensor a_shape_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({2}));
        auto a_shape_flat = a_shape_tensor.flat<int64_t>();
        a_shape_flat(0) = rows;
        a_shape_flat(1) = cols;
        
        // Create b tensor [cols, b_cols] or [b_cols, cols] depending on adjoint_b
        int b_rows = adjoint_b ? b_cols : cols;
        int b_actual_cols = adjoint_b ? cols : b_cols;
        tensorflow::Tensor b_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({b_rows, b_actual_cols}));
        auto b_flat = b_tensor.flat<float>();
        
        size_t remaining = size - offset;
        size_t b_size = b_rows * b_actual_cols * sizeof(float);
        if (remaining < b_size) {
            // Fill with default values if not enough data
            for (int i = 0; i < b_rows * b_actual_cols; i++) {
                b_flat(i) = 1.0f;
            }
        } else {
            for (int i = 0; i < b_rows * b_actual_cols; i++) {
                b_flat(i) = reinterpret_cast<const float*>(data + offset)[i];
            }
        }
        
        // Create input nodes
        auto a_indices_node = tensorflow::ops::Const(root, a_indices_tensor);
        auto a_values_node = tensorflow::ops::Const(root, a_values_tensor);
        auto a_shape_node = tensorflow::ops::Const(root, a_shape_tensor);
        auto b_node = tensorflow::ops::Const(root, b_tensor);
        
        // Create SparseTensorDenseMatMul operation
        auto sparse_matmul = tensorflow::ops::SparseTensorDenseMatMul(
            root, 
            a_indices_node, 
            a_values_node, 
            a_shape_node, 
            b_node,
            tensorflow::ops::SparseTensorDenseMatMul::Attrs()
                .AdjointA(adjoint_a)
                .AdjointB(adjoint_b)
        );
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({sparse_matmul}, &outputs);
        
        if (!status.ok()) {
            std::cout << "Operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output shape
        if (!outputs.empty()) {
            auto output_shape = outputs[0].shape();
            int expected_rows = adjoint_a ? cols : rows;
            int expected_cols = adjoint_b ? cols : b_cols;
            
            if (output_shape.dim_size(0) != expected_rows || 
                output_shape.dim_size(1) != expected_cols) {
                std::cout << "Unexpected output shape" << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}