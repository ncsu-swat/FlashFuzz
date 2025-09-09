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
        
        // Extract error_msg length
        int32_t msg_len;
        memcpy(&msg_len, data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        
        // Clamp message length to reasonable bounds
        msg_len = std::abs(msg_len) % 1024;
        
        if (offset + msg_len > size) {
            msg_len = size - offset;
        }
        
        // Extract error message
        std::string error_msg;
        if (msg_len > 0 && offset < size) {
            error_msg = std::string(reinterpret_cast<const char*>(data + offset), 
                                  std::min(static_cast<size_t>(msg_len), size - offset));
        } else {
            error_msg = "Fuzz test abort";
        }
        
        // Create a simple graph with Abort operation
        tensorflow::GraphDef graph_def;
        auto* node = graph_def.add_node();
        node->set_name("abort_op");
        node->set_op("Abort");
        
        // Set the error_msg attribute
        auto* attr = node->mutable_attr();
        (*attr)["error_msg"].set_s(error_msg);
        
        // Create session options
        tensorflow::SessionOptions options;
        
        // Create session
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        if (!session) {
            return 0;
        }
        
        // Create the session with the graph
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run the abort operation - this should cause an abort/error
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({}, {}, {"abort_op"}, &outputs);
        
        // The Abort operation should always fail, so we expect an error status
        if (status.ok()) {
            // This shouldn't happen with Abort op, but handle gracefully
            return 0;
        }
        
        // Close session
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}