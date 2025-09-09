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
        
        // Extract parameters from fuzzer input
        tensorflow::DataType dtype = static_cast<tensorflow::DataType>((data[offset] % 4) + 1); // DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64
        offset++;
        
        tensorflow::TensorShape shape_param;
        int32_t num_dims = (data[offset] % 4) + 1; // 1-4 dimensions
        offset++;
        
        for (int i = 0; i < num_dims && offset < size - 4; i++) {
            int32_t dim_size = 1 + (data[offset] % 10); // 1-10 size per dimension
            shape_param.AddDim(dim_size);
            offset++;
        }
        
        if (offset >= size - 4) return 0;
        
        std::string container = "test_container";
        std::string shared_name = "test_accumulator_" + std::to_string(data[offset]);
        offset++;
        
        // Create a simple TensorFlow session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Build the graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node = graph_def.add_node();
        
        node->set_name("sparse_conditional_accumulator");
        node->set_op("SparseConditionalAccumulator");
        
        // Set attributes
        tensorflow::AttrValue dtype_attr;
        dtype_attr.set_type(dtype);
        (*node->mutable_attr())["dtype"] = dtype_attr;
        
        tensorflow::AttrValue shape_attr;
        shape_param.AsProto(shape_attr.mutable_shape());
        (*node->mutable_attr())["shape"] = shape_attr;
        
        tensorflow::AttrValue container_attr;
        container_attr.set_s(container);
        (*node->mutable_attr())["container"] = container_attr;
        
        tensorflow::AttrValue shared_name_attr;
        shared_name_attr.set_s(shared_name);
        (*node->mutable_attr())["shared_name"] = shared_name_attr;
        
        // Create the session and run
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({}, {"sparse_conditional_accumulator:0"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Successfully created accumulator handle
        }
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}