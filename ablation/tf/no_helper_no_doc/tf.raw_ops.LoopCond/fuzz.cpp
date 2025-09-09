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
        
        if (size < sizeof(bool)) {
            return 0;
        }
        
        // Extract boolean value from fuzzer input
        bool input_value;
        memcpy(&input_value, data + offset, sizeof(bool));
        offset += sizeof(bool);
        
        // Create TensorFlow session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create input tensor
        tensorflow::Tensor input_tensor(tensorflow::DT_BOOL, tensorflow::TensorShape({}));
        input_tensor.scalar<bool>()() = input_value;
        
        // Build graph with LoopCond operation
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* input_node = graph_def.add_node();
        input_node->set_name("input");
        input_node->set_op("Placeholder");
        (*input_node->mutable_attr())["dtype"].set_type(tensorflow::DT_BOOL);
        (*input_node->mutable_attr())["shape"].mutable_shape();
        
        tensorflow::NodeDef* loop_cond_node = graph_def.add_node();
        loop_cond_node->set_name("loop_cond");
        loop_cond_node->set_op("LoopCond");
        loop_cond_node->add_input("input");
        
        // Create session and run
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({{"input", input_tensor}}, {"loop_cond"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Verify output is boolean type
            if (outputs[0].dtype() == tensorflow::DT_BOOL) {
                bool output_value = outputs[0].scalar<bool>()();
                // LoopCond should pass through the input value
                if (output_value == input_value) {
                    // Test passed
                }
            }
        }
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}