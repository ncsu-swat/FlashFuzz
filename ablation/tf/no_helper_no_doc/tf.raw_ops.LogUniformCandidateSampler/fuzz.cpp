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
        
        if (size < 20) return 0;
        
        // Extract parameters from fuzzer input
        int64_t num_true = *reinterpret_cast<const int64_t*>(data + offset);
        offset += sizeof(int64_t);
        
        int64_t num_sampled = *reinterpret_cast<const int64_t*>(data + offset);
        offset += sizeof(int64_t);
        
        bool unique = *reinterpret_cast<const bool*>(data + offset);
        offset += sizeof(bool);
        
        int64_t range_max = *reinterpret_cast<const int64_t*>(data + offset);
        offset += sizeof(int64_t);
        
        // Clamp values to reasonable ranges
        num_true = std::max(1LL, std::min(num_true, 1000LL));
        num_sampled = std::max(1LL, std::min(num_sampled, 1000LL));
        range_max = std::max(num_sampled + 1, std::min(range_max, 10000LL));
        
        // Calculate remaining data for true_classes
        size_t remaining_size = size - offset;
        size_t true_classes_size = std::min(remaining_size / sizeof(int64_t), static_cast<size_t>(num_true));
        
        if (true_classes_size == 0) return 0;
        
        // Create true_classes tensor
        tensorflow::Tensor true_classes(tensorflow::DT_INT64, tensorflow::TensorShape({static_cast<int64_t>(true_classes_size)}));
        auto true_classes_flat = true_classes.flat<int64_t>();
        
        for (size_t i = 0; i < true_classes_size && offset + sizeof(int64_t) <= size; ++i) {
            int64_t val = *reinterpret_cast<const int64_t*>(data + offset);
            true_classes_flat(i) = std::max(0LL, std::min(val, range_max - 1));
            offset += sizeof(int64_t);
        }
        
        // Create session and graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node_def = graph_def.add_node();
        
        node_def->set_name("log_uniform_candidate_sampler");
        node_def->set_op("LogUniformCandidateSampler");
        node_def->add_input("true_classes:0");
        
        // Set attributes
        tensorflow::AttrValue num_true_attr;
        num_true_attr.set_i(num_true);
        (*node_def->mutable_attr())["num_true"] = num_true_attr;
        
        tensorflow::AttrValue num_sampled_attr;
        num_sampled_attr.set_i(num_sampled);
        (*node_def->mutable_attr())["num_sampled"] = num_sampled_attr;
        
        tensorflow::AttrValue unique_attr;
        unique_attr.set_b(unique);
        (*node_def->mutable_attr())["unique"] = unique_attr;
        
        tensorflow::AttrValue range_max_attr;
        range_max_attr.set_i(range_max);
        (*node_def->mutable_attr())["range_max"] = range_max_attr;
        
        // Add placeholder for true_classes input
        tensorflow::NodeDef* placeholder_def = graph_def.add_node();
        placeholder_def->set_name("true_classes");
        placeholder_def->set_op("Placeholder");
        tensorflow::AttrValue dtype_attr;
        dtype_attr.set_type(tensorflow::DT_INT64);
        (*placeholder_def->mutable_attr())["dtype"] = dtype_attr;
        
        // Create session
        tensorflow::SessionOptions session_options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(session_options));
        
        if (!session) return 0;
        
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) return 0;
        
        // Run the operation
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"true_classes:0", true_classes}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {
            "log_uniform_candidate_sampler:0",  // sampled_candidates
            "log_uniform_candidate_sampler:1",  // true_expected_count
            "log_uniform_candidate_sampler:2"   // sampled_expected_count
        };
        
        status = session->Run(inputs, output_names, {}, &outputs);
        
        // Clean up
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}