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
        
        // Extract basic parameters
        uint32_t num_a_indices = *reinterpret_cast<const uint32_t*>(data + offset) % 10 + 1;
        offset += 4;
        uint32_t num_b_indices = *reinterpret_cast<const uint32_t*>(data + offset) % 10 + 1;
        offset += 4;
        uint32_t rank = *reinterpret_cast<const uint32_t*>(data + offset) % 3 + 1;
        offset += 4;
        uint32_t dtype_idx = *reinterpret_cast<const uint32_t*>(data + offset) % 3;
        offset += 4;
        
        if (offset + (num_a_indices + num_b_indices) * rank * 8 + 
            (num_a_indices + num_b_indices) * 4 + rank * 8 > size) {
            return 0;
        }
        
        tensorflow::DataType dtype;
        switch (dtype_idx) {
            case 0: dtype = tensorflow::DT_FLOAT; break;
            case 1: dtype = tensorflow::DT_INT32; break;
            case 2: dtype = tensorflow::DT_DOUBLE; break;
            default: dtype = tensorflow::DT_FLOAT; break;
        }
        
        // Create shape tensor
        tensorflow::Tensor shape_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({static_cast<int64_t>(rank)}));
        auto shape_flat = shape_tensor.flat<int64_t>();
        for (int i = 0; i < rank; i++) {
            shape_flat(i) = *reinterpret_cast<const int64_t*>(data + offset) % 10 + 1;
            offset += 8;
        }
        
        // Create a_indices tensor
        tensorflow::Tensor a_indices_tensor(tensorflow::DT_INT64, 
            tensorflow::TensorShape({static_cast<int64_t>(num_a_indices), static_cast<int64_t>(rank)}));
        auto a_indices_matrix = a_indices_tensor.matrix<int64_t>();
        for (int i = 0; i < num_a_indices; i++) {
            for (int j = 0; j < rank; j++) {
                a_indices_matrix(i, j) = *reinterpret_cast<const int64_t*>(data + offset) % shape_flat(j);
                offset += 8;
            }
        }
        
        // Create b_indices tensor
        tensorflow::Tensor b_indices_tensor(tensorflow::DT_INT64, 
            tensorflow::TensorShape({static_cast<int64_t>(num_b_indices), static_cast<int64_t>(rank)}));
        auto b_indices_matrix = b_indices_tensor.matrix<int64_t>();
        for (int i = 0; i < num_b_indices; i++) {
            for (int j = 0; j < rank; j++) {
                b_indices_matrix(i, j) = *reinterpret_cast<const int64_t*>(data + offset) % shape_flat(j);
                offset += 8;
            }
        }
        
        // Create a_values tensor
        tensorflow::Tensor a_values_tensor(dtype, tensorflow::TensorShape({static_cast<int64_t>(num_a_indices)}));
        if (dtype == tensorflow::DT_FLOAT) {
            auto a_values_flat = a_values_tensor.flat<float>();
            for (int i = 0; i < num_a_indices; i++) {
                a_values_flat(i) = *reinterpret_cast<const float*>(data + offset);
                offset += 4;
            }
        } else if (dtype == tensorflow::DT_INT32) {
            auto a_values_flat = a_values_tensor.flat<int32_t>();
            for (int i = 0; i < num_a_indices; i++) {
                a_values_flat(i) = *reinterpret_cast<const int32_t*>(data + offset);
                offset += 4;
            }
        } else if (dtype == tensorflow::DT_DOUBLE) {
            auto a_values_flat = a_values_tensor.flat<double>();
            for (int i = 0; i < num_a_indices; i++) {
                a_values_flat(i) = *reinterpret_cast<const double*>(data + offset);
                offset += 8;
            }
        }
        
        // Create b_values tensor
        tensorflow::Tensor b_values_tensor(dtype, tensorflow::TensorShape({static_cast<int64_t>(num_b_indices)}));
        if (dtype == tensorflow::DT_FLOAT) {
            auto b_values_flat = b_values_tensor.flat<float>();
            for (int i = 0; i < num_b_indices; i++) {
                b_values_flat(i) = *reinterpret_cast<const float*>(data + offset);
                offset += 4;
            }
        } else if (dtype == tensorflow::DT_INT32) {
            auto b_values_flat = b_values_tensor.flat<int32_t>();
            for (int i = 0; i < num_b_indices; i++) {
                b_values_flat(i) = *reinterpret_cast<const int32_t*>(data + offset);
                offset += 4;
            }
        } else if (dtype == tensorflow::DT_DOUBLE) {
            auto b_values_flat = b_values_tensor.flat<double>();
            for (int i = 0; i < num_b_indices; i++) {
                b_values_flat(i) = *reinterpret_cast<const double*>(data + offset);
                offset += 8;
            }
        }
        
        // Create TensorFlow scope and session
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input placeholders
        auto a_indices_ph = tensorflow::ops::Placeholder(root.WithOpName("a_indices"), tensorflow::DT_INT64);
        auto a_values_ph = tensorflow::ops::Placeholder(root.WithOpName("a_values"), dtype);
        auto a_shape_ph = tensorflow::ops::Placeholder(root.WithOpName("a_shape"), tensorflow::DT_INT64);
        auto b_indices_ph = tensorflow::ops::Placeholder(root.WithOpName("b_indices"), tensorflow::DT_INT64);
        auto b_values_ph = tensorflow::ops::Placeholder(root.WithOpName("b_values"), dtype);
        auto b_shape_ph = tensorflow::ops::Placeholder(root.WithOpName("b_shape"), tensorflow::DT_INT64);
        
        // Create SparseSparseMaximum operation
        auto sparse_max = tensorflow::ops::SparseSparseMaximum(root.WithOpName("sparse_max"),
            a_indices_ph, a_values_ph, a_shape_ph,
            b_indices_ph, b_values_ph, b_shape_ph);
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({
            {a_indices_ph, a_indices_tensor},
            {a_values_ph, a_values_tensor},
            {a_shape_ph, shape_tensor},
            {b_indices_ph, b_indices_tensor},
            {b_values_ph, b_values_tensor},
            {b_shape_ph, shape_tensor}
        }, {sparse_max.output_indices, sparse_max.output_values}, &outputs);
        
        if (!status.ok()) {
            std::cout << "Operation failed: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}