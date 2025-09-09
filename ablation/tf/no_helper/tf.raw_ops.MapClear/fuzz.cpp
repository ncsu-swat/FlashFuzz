#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/data_flow_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract parameters from fuzz input
        int32_t capacity = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        capacity = std::abs(capacity) % 1000; // Keep reasonable bounds
        
        int32_t memory_limit = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        memory_limit = std::abs(memory_limit) % 1000; // Keep reasonable bounds
        
        uint32_t container_len = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += sizeof(uint32_t);
        container_len = container_len % 100; // Limit string length
        
        uint32_t shared_name_len = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += sizeof(uint32_t);
        shared_name_len = shared_name_len % 100; // Limit string length
        
        if (offset + container_len + shared_name_len > size) return 0;
        
        std::string container(reinterpret_cast<const char*>(data + offset), container_len);
        offset += container_len;
        
        std::string shared_name(reinterpret_cast<const char*>(data + offset), shared_name_len);
        offset += shared_name_len;
        
        // Create dtypes list - use remaining data to determine types
        std::vector<tensorflow::DataType> dtypes;
        while (offset < size && dtypes.size() < 10) { // Limit number of dtypes
            uint8_t type_val = data[offset++];
            tensorflow::DataType dt = static_cast<tensorflow::DataType>(type_val % 23); // TF has ~23 basic types
            dtypes.push_back(dt);
        }
        
        if (dtypes.empty()) {
            dtypes.push_back(tensorflow::DT_FLOAT);
        }
        
        // Create TensorFlow scope and session
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create MapClear operation
        auto map_clear = tensorflow::ops::MapClear(
            root,
            dtypes,
            tensorflow::ops::MapClear::Capacity(capacity)
                .MemoryLimit(memory_limit)
                .Container(container)
                .SharedName(shared_name)
        );
        
        // Create session and run the operation
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({map_clear}, &outputs);
        
        // Check if operation completed (may fail due to invalid parameters, which is fine)
        if (!status.ok()) {
            // This is expected for many fuzz inputs with invalid parameters
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}