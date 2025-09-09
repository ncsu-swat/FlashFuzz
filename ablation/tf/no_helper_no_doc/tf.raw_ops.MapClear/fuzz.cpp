#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/node_def_builder.h>
#include <tensorflow/core/framework/fake_input.h>
#include <tensorflow/core/kernels/ops_testutil.h>
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < sizeof(int64_t)) {
            return 0;
        }
        
        // Extract capacity from fuzz input
        int64_t capacity;
        memcpy(&capacity, data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
        
        // Clamp capacity to reasonable bounds
        capacity = std::max(int64_t(1), std::min(capacity, int64_t(1000)));
        
        // Create a TensorFlow session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create graph definition
        tensorflow::GraphDef graph_def;
        
        // Add MapDataset operation to create a map
        auto* map_dataset_node = graph_def.add_node();
        map_dataset_node->set_name("map_dataset");
        map_dataset_node->set_op("MapDataset");
        
        // Add MapClear operation
        auto* map_clear_node = graph_def.add_node();
        map_clear_node->set_name("map_clear");
        map_clear_node->set_op("MapClear");
        
        // Set attributes for MapClear
        auto* capacity_attr = map_clear_node->mutable_attr();
        (*capacity_attr)["capacity"].set_i(capacity);
        
        // Set memory_limit attribute if we have more data
        if (offset + sizeof(int64_t) <= size) {
            int64_t memory_limit;
            memcpy(&memory_limit, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            memory_limit = std::max(int64_t(-1), std::min(memory_limit, int64_t(1000000)));
            (*capacity_attr)["memory_limit"].set_i(memory_limit);
        } else {
            (*capacity_attr)["memory_limit"].set_i(-1);
        }
        
        // Set container attribute if we have more data
        if (offset < size) {
            size_t container_len = std::min(size - offset, size_t(100));
            std::string container(reinterpret_cast<const char*>(data + offset), container_len);
            // Clean the string to avoid invalid characters
            for (char& c : container) {
                if (c < 32 || c > 126) c = 'a';
            }
            (*capacity_attr)["container"].set_s(container);
            offset += container_len;
        } else {
            (*capacity_attr)["container"].set_s("");
        }
        
        // Set shared_name attribute if we have more data
        if (offset < size) {
            size_t shared_name_len = std::min(size - offset, size_t(100));
            std::string shared_name(reinterpret_cast<const char*>(data + offset), shared_name_len);
            // Clean the string to avoid invalid characters
            for (char& c : shared_name) {
                if (c < 32 || c > 126) c = 'a';
            }
            (*capacity_attr)["shared_name"].set_s(shared_name);
        } else {
            (*capacity_attr)["shared_name"].set_s("");
        }
        
        // Create the graph
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run the MapClear operation
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({}, {"map_clear"}, {}, &outputs);
        
        // Clean up
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}