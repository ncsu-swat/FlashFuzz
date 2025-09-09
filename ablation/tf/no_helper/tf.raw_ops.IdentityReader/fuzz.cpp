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
#include <tensorflow/core/kernels/ops_util.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 2) {
            return 0;
        }
        
        // Extract container string length
        uint8_t container_len = data[offset++] % 32;  // Limit container length
        if (offset + container_len > size) {
            return 0;
        }
        
        std::string container(reinterpret_cast<const char*>(data + offset), container_len);
        offset += container_len;
        
        if (offset >= size) {
            return 0;
        }
        
        // Extract shared_name string length
        uint8_t shared_name_len = data[offset++] % 32;  // Limit shared_name length
        if (offset + shared_name_len > size) {
            return 0;
        }
        
        std::string shared_name(reinterpret_cast<const char*>(data + offset), shared_name_len);
        offset += shared_name_len;
        
        // Create a session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        if (!session) {
            return 0;
        }
        
        // Create GraphDef
        tensorflow::GraphDef graph_def;
        tensorflow::GraphDefBuilder builder(tensorflow::GraphDefBuilder::kFailImmediately);
        
        // Create IdentityReader node
        auto identity_reader_node = tensorflow::ops::SourceOp("IdentityReader", 
            builder.opts()
                .WithName("identity_reader")
                .WithAttr("container", container)
                .WithAttr("shared_name", shared_name));
        
        tensorflow::TF_RETURN_IF_ERROR(builder.ToGraphDef(&graph_def));
        
        // Create the session with the graph
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run the session to create the IdentityReader
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({}, {"identity_reader:0"}, {}, &outputs);
        
        // Clean up
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}