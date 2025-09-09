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
        
        if (size < 16) return 0;
        
        // Extract basic parameters
        uint32_t num_indices = *reinterpret_cast<const uint32_t*>(data + offset) % 100 + 1;
        offset += 4;
        uint32_t num_values = *reinterpret_cast<const uint32_t*>(data + offset) % 100 + 1;
        offset += 4;
        uint32_t rank = *reinterpret_cast<const uint32_t*>(data + offset) % 5 + 2; // rank >= 2
        offset += 4;
        uint32_t batch_size = *reinterpret_cast<const uint32_t*>(data + offset) % 10 + 1;
        offset += 4;
        
        if (offset + num_indices * 2 * 8 + num_values * 4 + rank * 8 > size) return 0;
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create sparse_indices tensor (2D: [num_indices, rank])
        tensorflow::Tensor sparse_indices_tensor(tensorflow::DT_INT64, 
            tensorflow::TensorShape({static_cast<int64_t>(num_indices), static_cast<int64_t>(rank)}));
        auto sparse_indices_matrix = sparse_indices_tensor.matrix<int64_t>();
        
        for (int i = 0; i < num_indices && offset + 8 <= size; ++i) {
            // First dimension should be batch index [0, batch_size)
            sparse_indices_matrix(i, 0) = (*reinterpret_cast<const uint64_t*>(data + offset)) % batch_size;
            offset += 8;
            
            for (int j = 1; j < rank && offset + 8 <= size; ++j) {
                sparse_indices_matrix(i, j) = *reinterpret_cast<const int64_t*>(data + offset) % 100;
                offset += 8;
            }
        }
        
        // Create sparse_values tensor (1D: [num_values])
        tensorflow::Tensor sparse_values_tensor(tensorflow::DT_FLOAT, 
            tensorflow::TensorShape({static_cast<int64_t>(num_values)}));
        auto sparse_values_flat = sparse_values_tensor.flat<float>();
        
        for (int i = 0; i < num_values && offset + 4 <= size; ++i) {
            sparse_values_flat(i) = *reinterpret_cast<const float*>(data + offset);
            offset += 4;
        }
        
        // Create sparse_shape tensor (1D: [rank])
        tensorflow::Tensor sparse_shape_tensor(tensorflow::DT_INT64, 
            tensorflow::TensorShape({static_cast<int64_t>(rank)}));
        auto sparse_shape_flat = sparse_shape_tensor.flat<int64_t>();
        
        // First dimension is batch size
        sparse_shape_flat(0) = batch_size;
        for (int i = 1; i < rank && offset + 8 <= size; ++i) {
            sparse_shape_flat(i) = (*reinterpret_cast<const uint64_t*>(data + offset)) % 100 + 1;
            offset += 8;
        }
        
        // Create input operations
        auto sparse_indices_op = tensorflow::ops::Const(root, sparse_indices_tensor);
        auto sparse_values_op = tensorflow::ops::Const(root, sparse_values_tensor);
        auto sparse_shape_op = tensorflow::ops::Const(root, sparse_shape_tensor);
        
        // Extract container and shared_name strings
        std::string container = "";
        std::string shared_name = "";
        
        if (offset + 2 <= size) {
            uint8_t container_len = data[offset] % 10;
            offset++;
            if (offset + container_len <= size) {
                container = std::string(reinterpret_cast<const char*>(data + offset), container_len);
                offset += container_len;
            }
        }
        
        if (offset + 2 <= size) {
            uint8_t shared_name_len = data[offset] % 10;
            offset++;
            if (offset + shared_name_len <= size) {
                shared_name = std::string(reinterpret_cast<const char*>(data + offset), shared_name_len);
                offset += shared_name_len;
            }
        }
        
        // Create AddManySparseToTensorsMap operation
        auto add_op = tensorflow::ops::AddManySparseToTensorsMap(
            root,
            sparse_indices_op,
            sparse_values_op,
            sparse_shape_op,
            tensorflow::ops::AddManySparseToTensorsMap::Container(container)
                .SharedName(shared_name)
        );
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({add_op}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Verify output tensor properties
            const auto& output = outputs[0];
            if (output.dtype() == tensorflow::DT_INT64 && 
                output.dims() == 1 && 
                output.dim_size(0) == batch_size) {
                // Success - output has expected shape and type
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}