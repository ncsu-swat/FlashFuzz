#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/str_util.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/common_runtime/direct_session.h>
#include <tensorflow/core/framework/node_def_builder.h>
#include <tensorflow/core/framework/op_def_builder.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract fuzzer inputs
        int32_t num_elements_val = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        
        bool allow_small_batch = (data[offset] % 2) == 1;
        offset += 1;
        
        bool wait_for_incomplete = (data[offset] % 2) == 1;
        offset += 1;
        
        int32_t timeout_ms = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        
        uint8_t num_component_types = data[offset] % 4 + 1; // 1-4 component types
        offset += 1;
        
        if (offset + num_component_types > size) return 0;
        
        // Clamp num_elements to reasonable range
        num_elements_val = std::abs(num_elements_val) % 100 + 1;
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create barrier handle tensor (mutable string)
        tensorflow::Tensor handle_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        handle_tensor.scalar<tensorflow::tstring>()() = "test_barrier_handle";
        
        // Create num_elements tensor
        tensorflow::Tensor num_elements_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        num_elements_tensor.scalar<int32_t>()() = num_elements_val;
        
        // Create component types list
        std::vector<tensorflow::DataType> component_types;
        for (int i = 0; i < num_component_types; i++) {
            uint8_t type_idx = data[offset + i] % 4;
            switch (type_idx) {
                case 0: component_types.push_back(tensorflow::DT_FLOAT); break;
                case 1: component_types.push_back(tensorflow::DT_INT32); break;
                case 2: component_types.push_back(tensorflow::DT_STRING); break;
                case 3: component_types.push_back(tensorflow::DT_DOUBLE); break;
            }
        }
        
        // Build graph with BarrierTakeMany operation
        tensorflow::GraphDefBuilder builder(tensorflow::GraphDefBuilder::kFailImmediately);
        
        // Add placeholder nodes for inputs
        auto handle_node = tensorflow::ops::Placeholder(builder.opts()
            .WithName("handle")
            .WithAttr("dtype", tensorflow::DT_STRING));
            
        auto num_elements_node = tensorflow::ops::Placeholder(builder.opts()
            .WithName("num_elements")
            .WithAttr("dtype", tensorflow::DT_INT32));
        
        // Create BarrierTakeMany node
        tensorflow::NodeDefBuilder barrier_take_many_builder("barrier_take_many", "BarrierTakeMany");
        barrier_take_many_builder.Input(handle_node.name(), 0, tensorflow::DT_STRING);
        barrier_take_many_builder.Input(num_elements_node.name(), 0, tensorflow::DT_INT32);
        barrier_take_many_builder.Attr("component_types", component_types);
        barrier_take_many_builder.Attr("allow_small_batch", allow_small_batch);
        barrier_take_many_builder.Attr("wait_for_incomplete", wait_for_incomplete);
        barrier_take_many_builder.Attr("timeout_ms", static_cast<int64_t>(timeout_ms));
        
        tensorflow::NodeDef barrier_take_many_def;
        auto status = barrier_take_many_builder.Finalize(&barrier_take_many_def);
        if (!status.ok()) {
            return 0;
        }
        
        tensorflow::GraphDef graph_def;
        status = builder.ToGraphDef(&graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Add the BarrierTakeMany node to graph
        *graph_def.add_node() = barrier_take_many_def;
        
        // Create session and add graph
        status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Prepare inputs
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"handle:0", handle_tensor},
            {"num_elements:0", num_elements_tensor}
        };
        
        // Prepare output names
        std::vector<std::string> output_names = {
            "barrier_take_many:0", // indices
            "barrier_take_many:1", // keys
        };
        
        // Add value outputs based on component types
        for (size_t i = 0; i < component_types.size(); i++) {
            output_names.push_back("barrier_take_many:" + std::to_string(i + 2));
        }
        
        std::vector<tensorflow::Tensor> outputs;
        
        // Run the operation (this will likely fail since we don't have a real barrier,
        // but we're testing the operation construction and parameter validation)
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