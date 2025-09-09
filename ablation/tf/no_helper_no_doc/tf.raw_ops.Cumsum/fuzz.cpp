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
        
        // Extract parameters from fuzz input
        int32_t rank = (data[offset] % 4) + 1; // 1-4 dimensions
        offset++;
        
        bool exclusive = data[offset] % 2;
        offset++;
        
        bool reverse = data[offset] % 2;
        offset++;
        
        // Create tensor shape
        tensorflow::TensorShape input_shape;
        tensorflow::TensorShape axis_shape;
        
        for (int i = 0; i < rank && offset < size; i++) {
            int32_t dim_size = (data[offset] % 10) + 1; // 1-10 size per dimension
            input_shape.AddDim(dim_size);
            offset++;
        }
        
        if (offset >= size) return 0;
        
        // Choose axis
        int32_t axis_val = (data[offset] % rank);
        if (data[offset + 1] % 2) axis_val = -axis_val - 1; // negative axis
        offset += 2;
        
        // Choose data type
        tensorflow::DataType dtype;
        switch (data[offset] % 4) {
            case 0: dtype = tensorflow::DT_INT32; break;
            case 1: dtype = tensorflow::DT_INT64; break;
            case 2: dtype = tensorflow::DT_FLOAT; break;
            case 3: dtype = tensorflow::DT_DOUBLE; break;
        }
        offset++;
        
        // Create input tensor
        tensorflow::Tensor input_tensor(dtype, input_shape);
        tensorflow::Tensor axis_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        axis_tensor.scalar<int32_t>()() = axis_val;
        
        // Fill input tensor with fuzz data
        int64_t num_elements = input_tensor.NumElements();
        
        if (dtype == tensorflow::DT_INT32) {
            auto flat = input_tensor.flat<int32_t>();
            for (int64_t i = 0; i < num_elements && offset < size; i++) {
                flat(i) = static_cast<int32_t>(data[offset]);
                offset++;
            }
        } else if (dtype == tensorflow::DT_INT64) {
            auto flat = input_tensor.flat<int64_t>();
            for (int64_t i = 0; i < num_elements && offset < size; i++) {
                flat(i) = static_cast<int64_t>(data[offset]);
                offset++;
            }
        } else if (dtype == tensorflow::DT_FLOAT) {
            auto flat = input_tensor.flat<float>();
            for (int64_t i = 0; i < num_elements && offset < size; i++) {
                flat(i) = static_cast<float>(data[offset]) / 255.0f;
                offset++;
            }
        } else if (dtype == tensorflow::DT_DOUBLE) {
            auto flat = input_tensor.flat<double>();
            for (int64_t i = 0; i < num_elements && offset < size; i++) {
                flat(i) = static_cast<double>(data[offset]) / 255.0;
                offset++;
            }
        }
        
        // Create session and graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* input_node = graph_def.add_node();
        input_node->set_name("input");
        input_node->set_op("Placeholder");
        (*input_node->mutable_attr())["dtype"].set_type(dtype);
        
        tensorflow::NodeDef* axis_node = graph_def.add_node();
        axis_node->set_name("axis");
        axis_node->set_op("Placeholder");
        (*axis_node->mutable_attr())["dtype"].set_type(tensorflow::DT_INT32);
        
        tensorflow::NodeDef* cumsum_node = graph_def.add_node();
        cumsum_node->set_name("cumsum");
        cumsum_node->set_op("Cumsum");
        cumsum_node->add_input("input");
        cumsum_node->add_input("axis");
        (*cumsum_node->mutable_attr())["T"].set_type(dtype);
        (*cumsum_node->mutable_attr())["Tidx"].set_type(tensorflow::DT_INT32);
        (*cumsum_node->mutable_attr())["exclusive"].set_b(exclusive);
        (*cumsum_node->mutable_attr())["reverse"].set_b(reverse);
        
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({{"input", input_tensor}, {"axis", axis_tensor}}, 
                             {"cumsum"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Operation succeeded, verify output shape matches input shape
            if (outputs[0].shape() == input_shape) {
                // Basic validation passed
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