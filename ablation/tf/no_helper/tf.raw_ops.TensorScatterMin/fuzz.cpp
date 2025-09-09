#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions and parameters from fuzz input
        uint32_t tensor_dim1 = (data[offset] % 10) + 1;
        uint32_t tensor_dim2 = (data[offset + 1] % 10) + 1;
        uint32_t num_indices = (data[offset + 2] % 5) + 1;
        uint32_t data_type = data[offset + 3] % 3; // 0: float, 1: int32, 2: int64
        offset += 4;
        
        if (offset + tensor_dim1 * tensor_dim2 * 4 + num_indices * 8 + num_indices * 4 > size) {
            return 0;
        }
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create tensor shape
        tensorflow::TensorShape tensor_shape({static_cast<int64_t>(tensor_dim1), static_cast<int64_t>(tensor_dim2)});
        tensorflow::TensorShape indices_shape({static_cast<int64_t>(num_indices), 2});
        tensorflow::TensorShape updates_shape({static_cast<int64_t>(num_indices)});
        
        if (data_type == 0) { // float
            // Create input tensor
            tensorflow::Tensor tensor(tensorflow::DT_FLOAT, tensor_shape);
            auto tensor_flat = tensor.flat<float>();
            for (int i = 0; i < tensor_dim1 * tensor_dim2 && offset + 4 <= size; ++i) {
                float val;
                memcpy(&val, data + offset, sizeof(float));
                tensor_flat(i) = val;
                offset += 4;
            }
            
            // Create indices tensor
            tensorflow::Tensor indices(tensorflow::DT_INT32, indices_shape);
            auto indices_flat = indices.flat<int32_t>();
            for (int i = 0; i < num_indices * 2 && offset + 4 <= size; ++i) {
                int32_t val;
                memcpy(&val, data + offset, sizeof(int32_t));
                indices_flat(i) = abs(val) % (i % 2 == 0 ? tensor_dim1 : tensor_dim2);
                offset += 4;
            }
            
            // Create updates tensor
            tensorflow::Tensor updates(tensorflow::DT_FLOAT, updates_shape);
            auto updates_flat = updates.flat<float>();
            for (int i = 0; i < num_indices && offset + 4 <= size; ++i) {
                float val;
                memcpy(&val, data + offset, sizeof(float));
                updates_flat(i) = val;
                offset += 4;
            }
            
            // Create operation
            auto tensor_scatter_min = tensorflow::ops::TensorScatterMin(root, tensor, indices, updates);
            
            // Execute
            tensorflow::ClientSession session(root);
            std::vector<tensorflow::Tensor> outputs;
            session.Run({tensor_scatter_min}, &outputs);
            
        } else if (data_type == 1) { // int32
            // Create input tensor
            tensorflow::Tensor tensor(tensorflow::DT_INT32, tensor_shape);
            auto tensor_flat = tensor.flat<int32_t>();
            for (int i = 0; i < tensor_dim1 * tensor_dim2 && offset + 4 <= size; ++i) {
                int32_t val;
                memcpy(&val, data + offset, sizeof(int32_t));
                tensor_flat(i) = val;
                offset += 4;
            }
            
            // Create indices tensor
            tensorflow::Tensor indices(tensorflow::DT_INT32, indices_shape);
            auto indices_flat = indices.flat<int32_t>();
            for (int i = 0; i < num_indices * 2 && offset + 4 <= size; ++i) {
                int32_t val;
                memcpy(&val, data + offset, sizeof(int32_t));
                indices_flat(i) = abs(val) % (i % 2 == 0 ? tensor_dim1 : tensor_dim2);
                offset += 4;
            }
            
            // Create updates tensor
            tensorflow::Tensor updates(tensorflow::DT_INT32, updates_shape);
            auto updates_flat = updates.flat<int32_t>();
            for (int i = 0; i < num_indices && offset + 4 <= size; ++i) {
                int32_t val;
                memcpy(&val, data + offset, sizeof(int32_t));
                updates_flat(i) = val;
                offset += 4;
            }
            
            // Create operation
            auto tensor_scatter_min = tensorflow::ops::TensorScatterMin(root, tensor, indices, updates);
            
            // Execute
            tensorflow::ClientSession session(root);
            std::vector<tensorflow::Tensor> outputs;
            session.Run({tensor_scatter_min}, &outputs);
            
        } else { // int64
            // Create input tensor
            tensorflow::Tensor tensor(tensorflow::DT_INT64, tensor_shape);
            auto tensor_flat = tensor.flat<int64_t>();
            for (int i = 0; i < tensor_dim1 * tensor_dim2 && offset + 8 <= size; ++i) {
                int64_t val;
                memcpy(&val, data + offset, sizeof(int64_t));
                tensor_flat(i) = val;
                offset += 8;
            }
            
            // Create indices tensor
            tensorflow::Tensor indices(tensorflow::DT_INT64, indices_shape);
            auto indices_flat = indices.flat<int64_t>();
            for (int i = 0; i < num_indices * 2 && offset + 8 <= size; ++i) {
                int64_t val;
                memcpy(&val, data + offset, sizeof(int64_t));
                indices_flat(i) = abs(val) % (i % 2 == 0 ? tensor_dim1 : tensor_dim2);
                offset += 8;
            }
            
            // Create updates tensor
            tensorflow::Tensor updates(tensorflow::DT_INT64, updates_shape);
            auto updates_flat = updates.flat<int64_t>();
            for (int i = 0; i < num_indices && offset + 8 <= size; ++i) {
                int64_t val;
                memcpy(&val, data + offset, sizeof(int64_t));
                updates_flat(i) = val;
                offset += 8;
            }
            
            // Create operation
            auto tensor_scatter_min = tensorflow::ops::TensorScatterMin(root, tensor, indices, updates);
            
            // Execute
            tensorflow::ClientSession session(root);
            std::vector<tensorflow::Tensor> outputs;
            session.Run({tensor_scatter_min}, &outputs);
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}