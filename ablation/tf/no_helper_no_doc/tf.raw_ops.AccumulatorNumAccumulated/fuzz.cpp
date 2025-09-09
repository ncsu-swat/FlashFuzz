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
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/kernels/ops_util.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < sizeof(int32_t)) {
            return 0;
        }
        
        // Extract handle string length
        int32_t handle_len;
        memcpy(&handle_len, data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        
        // Clamp handle length to reasonable bounds
        handle_len = std::max(0, std::min(handle_len, static_cast<int32_t>(size - offset)));
        
        if (offset + handle_len > size) {
            return 0;
        }
        
        // Create handle string
        std::string handle_str(reinterpret_cast<const char*>(data + offset), handle_len);
        offset += handle_len;
        
        // Create TensorFlow session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create graph definition
        tensorflow::GraphDef graph_def;
        tensorflow::GraphDefBuilder builder(tensorflow::GraphDefBuilder::kFailImmediately);
        
        // Create a constant node for the handle
        auto handle_node = tensorflow::ops::Const(handle_str, builder.opts().WithName("handle"));
        
        // Create AccumulatorNumAccumulated operation
        auto accumulator_op = tensorflow::ops::UnaryOp("AccumulatorNumAccumulated", handle_node,
                                                      builder.opts().WithName("accumulator_num_accumulated"));
        
        // Build the graph
        tensorflow::Status status = builder.ToGraphDef(&graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Create the session with the graph
        status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Prepare input tensors
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
        
        // Run the session
        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {"accumulator_num_accumulated:0"};
        
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