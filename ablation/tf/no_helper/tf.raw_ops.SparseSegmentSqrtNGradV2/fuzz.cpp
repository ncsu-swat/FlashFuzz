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
        
        if (size < 20) return 0; // Need minimum data for parameters
        
        // Extract dimensions and parameters from fuzz input
        int32_t grad_dim0 = (data[offset] % 10) + 1;
        offset++;
        int32_t grad_dim1 = (data[offset] % 10) + 1;
        offset++;
        int32_t num_indices = (data[offset] % 10) + 1;
        offset++;
        int32_t num_segments = (data[offset] % 5) + 1;
        offset++;
        int32_t dense_output_dim0_val = (data[offset] % 20) + 1;
        offset++;
        
        // Determine data types from fuzz input
        tensorflow::DataType grad_dtype = tensorflow::DT_FLOAT;
        tensorflow::DataType indices_dtype = tensorflow::DT_INT32;
        
        if (offset < size) {
            switch (data[offset] % 4) {
                case 0: grad_dtype = tensorflow::DT_BFLOAT16; break;
                case 1: grad_dtype = tensorflow::DT_HALF; break;
                case 2: grad_dtype = tensorflow::DT_FLOAT; break;
                case 3: grad_dtype = tensorflow::DT_DOUBLE; break;
            }
            offset++;
        }
        
        if (offset < size) {
            indices_dtype = (data[offset] % 2 == 0) ? tensorflow::DT_INT32 : tensorflow::DT_INT64;
            offset++;
        }
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create grad tensor
        tensorflow::TensorShape grad_shape({grad_dim0, grad_dim1});
        tensorflow::Tensor grad_tensor(grad_dtype, grad_shape);
        
        // Fill grad tensor with fuzz data
        if (grad_dtype == tensorflow::DT_FLOAT) {
            auto grad_flat = grad_tensor.flat<float>();
            for (int i = 0; i < grad_flat.size() && offset < size; i++) {
                grad_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
                offset++;
            }
        }
        
        // Create indices tensor
        tensorflow::TensorShape indices_shape({num_indices});
        tensorflow::Tensor indices_tensor(indices_dtype, indices_shape);
        
        if (indices_dtype == tensorflow::DT_INT32) {
            auto indices_flat = indices_tensor.flat<int32_t>();
            for (int i = 0; i < indices_flat.size(); i++) {
                indices_flat(i) = (offset < size ? data[offset] : i) % dense_output_dim0_val;
                offset++;
            }
        } else {
            auto indices_flat = indices_tensor.flat<int64_t>();
            for (int i = 0; i < indices_flat.size(); i++) {
                indices_flat(i) = (offset < size ? data[offset] : i) % dense_output_dim0_val;
                offset++;
            }
        }
        
        // Create segment_ids tensor
        tensorflow::TensorShape segment_ids_shape({num_indices});
        tensorflow::Tensor segment_ids_tensor(indices_dtype, segment_ids_shape);
        
        if (indices_dtype == tensorflow::DT_INT32) {
            auto segment_ids_flat = segment_ids_tensor.flat<int32_t>();
            for (int i = 0; i < segment_ids_flat.size(); i++) {
                segment_ids_flat(i) = (offset < size ? data[offset] : i) % num_segments;
                offset++;
            }
        } else {
            auto segment_ids_flat = segment_ids_tensor.flat<int64_t>();
            for (int i = 0; i < segment_ids_flat.size(); i++) {
                segment_ids_flat(i) = (offset < size ? data[offset] : i) % num_segments;
                offset++;
            }
        }
        
        // Create dense_output_dim0 tensor
        tensorflow::TensorShape dense_dim_shape({});
        tensorflow::Tensor dense_output_dim0_tensor(tensorflow::DT_INT32, dense_dim_shape);
        dense_output_dim0_tensor.scalar<int32_t>()() = dense_output_dim0_val;
        
        // Convert tensors to ops
        auto grad_op = tensorflow::ops::Const(root, grad_tensor);
        auto indices_op = tensorflow::ops::Const(root, indices_tensor);
        auto segment_ids_op = tensorflow::ops::Const(root, segment_ids_tensor);
        auto dense_output_dim0_op = tensorflow::ops::Const(root, dense_output_dim0_tensor);
        
        // Create SparseSegmentSqrtNGradV2 operation
        auto sparse_segment_sqrt_n_grad_v2 = tensorflow::ops::SparseSegmentSqrtNGradV2(
            root, grad_op, indices_op, segment_ids_op, dense_output_dim0_op);
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({sparse_segment_sqrt_n_grad_v2.output, 
                                                sparse_segment_sqrt_n_grad_v2.sorted_unique_indices}, 
                                               &outputs);
        
        if (!status.ok()) {
            std::cout << "Operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify outputs
        if (outputs.size() == 2) {
            const auto& output = outputs[0];
            const auto& sorted_unique_indices = outputs[1];
            
            // Basic validation
            if (output.dtype() == grad_dtype && 
                sorted_unique_indices.dtype() == indices_dtype) {
                // Success - operation completed with expected types
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}