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
        
        // Extract parameters from fuzzer input
        int32_t capacity = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        
        uint8_t num_component_types = data[offset] % 5; // Limit to reasonable number
        offset += 1;
        
        uint8_t num_shapes = data[offset] % 5; // Limit to reasonable number  
        offset += 1;
        
        uint8_t container_len = data[offset] % 32; // Limit container name length
        offset += 1;
        
        uint8_t shared_name_len = data[offset] % 32; // Limit shared name length
        offset += 1;
        
        if (offset + container_len + shared_name_len > size) return 0;
        
        // Create component types
        std::vector<tensorflow::DataType> component_types;
        for (int i = 0; i < num_component_types && offset < size; ++i) {
            tensorflow::DataType dt = static_cast<tensorflow::DataType>(data[offset] % 23 + 1); // Valid DT range
            component_types.push_back(dt);
            offset += 1;
        }
        
        // Create shapes
        std::vector<tensorflow::TensorShape> shapes;
        for (int i = 0; i < num_shapes && offset + 4 <= size; ++i) {
            int32_t dim_count = data[offset] % 4; // Limit dimensions
            offset += 1;
            
            std::vector<int64_t> dims;
            for (int j = 0; j < dim_count && offset + 4 <= size; ++j) {
                int32_t dim_size = *reinterpret_cast<const int32_t*>(data + offset);
                dim_size = std::abs(dim_size) % 100 + 1; // Positive dimensions
                dims.push_back(dim_size);
                offset += sizeof(int32_t);
            }
            
            if (!dims.empty()) {
                shapes.emplace_back(dims);
            } else {
                shapes.emplace_back(); // Empty shape
            }
        }
        
        // Extract container and shared_name
        std::string container;
        if (container_len > 0 && offset + container_len <= size) {
            container = std::string(reinterpret_cast<const char*>(data + offset), container_len);
            offset += container_len;
        }
        
        std::string shared_name;
        if (shared_name_len > 0 && offset + shared_name_len <= size) {
            shared_name = std::string(reinterpret_cast<const char*>(data + offset), shared_name_len);
            offset += shared_name_len;
        }
        
        // Create TensorFlow scope and session
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create PriorityQueue operation
        auto priority_queue_attrs = tensorflow::ops::PriorityQueue::Attrs()
            .ComponentTypes(component_types)
            .Capacity(capacity)
            .Container(container)
            .SharedName(shared_name);
            
        if (!shapes.empty()) {
            priority_queue_attrs = priority_queue_attrs.Shapes(shapes);
        }
        
        auto priority_queue = tensorflow::ops::PriorityQueue(root, priority_queue_attrs);
        
        // Create session and run the operation
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        auto status = session.Run({priority_queue.handle}, &outputs);
        
        if (!status.ok()) {
            std::cout << "PriorityQueue operation failed: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}