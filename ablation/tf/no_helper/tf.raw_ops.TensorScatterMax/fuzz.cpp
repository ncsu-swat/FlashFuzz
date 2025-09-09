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
        
        // Extract dimensions from fuzz data
        uint32_t tensor_size = (data[offset] % 16) + 1;
        offset += 1;
        
        uint32_t num_indices = (data[offset] % 8) + 1;
        offset += 1;
        
        if (offset + tensor_size * 4 + num_indices * 8 + num_indices * 4 > size) {
            return 0;
        }
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create tensor data
        std::vector<int32_t> tensor_data(tensor_size);
        for (uint32_t i = 0; i < tensor_size && offset + 4 <= size; ++i) {
            memcpy(&tensor_data[i], data + offset, sizeof(int32_t));
            offset += 4;
        }
        
        // Create indices data (2D: [num_indices, 1])
        std::vector<int32_t> indices_data(num_indices);
        for (uint32_t i = 0; i < num_indices && offset + 4 <= size; ++i) {
            indices_data[i] = (reinterpret_cast<const int32_t*>(data + offset)[0]) % tensor_size;
            if (indices_data[i] < 0) indices_data[i] = -indices_data[i];
            offset += 4;
        }
        
        // Create updates data
        std::vector<int32_t> updates_data(num_indices);
        for (uint32_t i = 0; i < num_indices && offset + 4 <= size; ++i) {
            memcpy(&updates_data[i], data + offset, sizeof(int32_t));
            offset += 4;
        }
        
        // Create TensorFlow tensors
        tensorflow::Tensor tensor(tensorflow::DT_INT32, tensorflow::TensorShape({static_cast<int64_t>(tensor_size)}));
        auto tensor_flat = tensor.flat<int32_t>();
        for (uint32_t i = 0; i < tensor_size; ++i) {
            tensor_flat(i) = tensor_data[i];
        }
        
        tensorflow::Tensor indices(tensorflow::DT_INT32, tensorflow::TensorShape({static_cast<int64_t>(num_indices), 1}));
        auto indices_matrix = indices.matrix<int32_t>();
        for (uint32_t i = 0; i < num_indices; ++i) {
            indices_matrix(i, 0) = indices_data[i];
        }
        
        tensorflow::Tensor updates(tensorflow::DT_INT32, tensorflow::TensorShape({static_cast<int64_t>(num_indices)}));
        auto updates_flat = updates.flat<int32_t>();
        for (uint32_t i = 0; i < num_indices; ++i) {
            updates_flat(i) = updates_data[i];
        }
        
        // Create TensorScatterMax operation
        auto tensor_input = tensorflow::ops::Const(root, tensor);
        auto indices_input = tensorflow::ops::Const(root, indices);
        auto updates_input = tensorflow::ops::Const(root, updates);
        
        auto scatter_max = tensorflow::ops::TensorScatterMax(root, tensor_input, indices_input, updates_input);
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({scatter_max}, &outputs);
        
        if (!status.ok()) {
            std::cout << "TensorFlow operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output shape matches input tensor shape
        if (outputs.size() > 0) {
            const tensorflow::Tensor& result = outputs[0];
            if (result.shape().dims() != tensor.shape().dims() || 
                result.shape().dim_size(0) != tensor.shape().dim_size(0)) {
                std::cout << "Output shape mismatch" << std::endl;
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