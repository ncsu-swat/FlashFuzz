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
        
        if (size < 16) return 0;
        
        // Extract capacity from fuzz data
        int32_t capacity = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        capacity = std::abs(capacity) % 1000 + 1; // Ensure positive capacity
        
        // Extract memory_limit from fuzz data
        int64_t memory_limit = *reinterpret_cast<const int64_t*>(data + offset);
        offset += sizeof(int64_t);
        memory_limit = std::abs(memory_limit) % 1000000; // Reasonable memory limit
        
        // Extract container and shared_name lengths
        if (offset + 8 > size) return 0;
        uint32_t container_len = *reinterpret_cast<const uint32_t*>(data + offset) % 100;
        offset += sizeof(uint32_t);
        uint32_t shared_name_len = *reinterpret_cast<const uint32_t*>(data + offset) % 100;
        offset += sizeof(uint32_t);
        
        // Extract container string
        std::string container = "";
        if (container_len > 0 && offset + container_len <= size) {
            container = std::string(reinterpret_cast<const char*>(data + offset), container_len);
            offset += container_len;
        }
        
        // Extract shared_name string
        std::string shared_name = "";
        if (shared_name_len > 0 && offset + shared_name_len <= size) {
            shared_name = std::string(reinterpret_cast<const char*>(data + offset), shared_name_len);
            offset += shared_name_len;
        }
        
        // Create a simple test session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create GraphDef
        tensorflow::GraphDef graph_def;
        
        // Add MapStage operation first to create a staging area
        auto* stage_node = graph_def.add_node();
        stage_node->set_name("map_stage");
        stage_node->set_op("MapStage");
        
        // Add input for MapStage (key)
        auto* key_node = graph_def.add_node();
        key_node->set_name("key");
        key_node->set_op("Const");
        auto* key_attr = key_node->mutable_attr();
        (*key_attr)["dtype"].set_type(tensorflow::DT_INT64);
        auto* key_tensor = (*key_attr)["value"].mutable_tensor();
        key_tensor->set_dtype(tensorflow::DT_INT64);
        key_tensor->mutable_tensor_shape();
        key_tensor->add_int64_val(1);
        
        // Add input for MapStage (values)
        auto* values_node = graph_def.add_node();
        values_node->set_name("values");
        values_node->set_op("Const");
        auto* values_attr = values_node->mutable_attr();
        (*values_attr)["dtype"].set_type(tensorflow::DT_FLOAT);
        auto* values_tensor = (*values_attr)["value"].mutable_tensor();
        values_tensor->set_dtype(tensorflow::DT_FLOAT);
        values_tensor->mutable_tensor_shape()->add_dim()->set_size(1);
        values_tensor->add_float_val(1.0f);
        
        stage_node->add_input("key");
        stage_node->add_input("values");
        
        auto* stage_attr = stage_node->mutable_attr();
        (*stage_attr)["capacity"].set_i(capacity);
        (*stage_attr)["memory_limit"].set_i(memory_limit);
        (*stage_attr)["container"].set_s(container);
        (*stage_attr)["shared_name"].set_s(shared_name);
        (*stage_attr)["dtypes"].mutable_list()->add_type(tensorflow::DT_FLOAT);
        
        // Add MapUnstage operation
        auto* unstage_node = graph_def.add_node();
        unstage_node->set_name("map_unstage");
        unstage_node->set_op("MapUnstage");
        
        // Add key input for MapUnstage
        auto* unstage_key_node = graph_def.add_node();
        unstage_key_node->set_name("unstage_key");
        unstage_key_node->set_op("Const");
        auto* unstage_key_attr = unstage_key_node->mutable_attr();
        (*unstage_key_attr)["dtype"].set_type(tensorflow::DT_INT64);
        auto* unstage_key_tensor = (*unstage_key_attr)["value"].mutable_tensor();
        unstage_key_tensor->set_dtype(tensorflow::DT_INT64);
        unstage_key_tensor->mutable_tensor_shape();
        unstage_key_tensor->add_int64_val(1);
        
        // Add indices input for MapUnstage
        auto* indices_node = graph_def.add_node();
        indices_node->set_name("indices");
        indices_node->set_op("Const");
        auto* indices_attr = indices_node->mutable_attr();
        (*indices_attr)["dtype"].set_type(tensorflow::DT_INT32);
        auto* indices_tensor = (*indices_attr)["value"].mutable_tensor();
        indices_tensor->set_dtype(tensorflow::DT_INT32);
        indices_tensor->mutable_tensor_shape()->add_dim()->set_size(1);
        indices_tensor->add_int_val(0);
        
        unstage_node->add_input("unstage_key");
        unstage_node->add_input("indices");
        
        auto* unstage_attr = unstage_node->mutable_attr();
        (*unstage_attr)["capacity"].set_i(capacity);
        (*unstage_attr)["memory_limit"].set_i(memory_limit);
        (*unstage_attr)["container"].set_s(container);
        (*unstage_attr)["shared_name"].set_s(shared_name);
        (*unstage_attr)["dtypes"].mutable_list()->add_type(tensorflow::DT_FLOAT);
        
        // Create the session and run
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run MapStage first
        std::vector<tensorflow::Tensor> stage_outputs;
        status = session->Run({}, {}, {"map_stage"}, &stage_outputs);
        
        // Run MapUnstage
        std::vector<tensorflow::Tensor> unstage_outputs;
        status = session->Run({}, {"map_unstage:0"}, {}, &unstage_outputs);
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}