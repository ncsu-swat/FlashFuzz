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
        
        // Extract dimensions and data type
        uint32_t rows = *reinterpret_cast<const uint32_t*>(data + offset) % 100 + 1;
        offset += 4;
        uint32_t cols = *reinterpret_cast<const uint32_t*>(data + offset) % 100 + 1;
        offset += 4;
        uint32_t i = *reinterpret_cast<const uint32_t*>(data + offset) % rows;
        offset += 4;
        uint32_t dtype_val = *reinterpret_cast<const uint32_t*>(data + offset) % 3;
        offset += 4;
        
        tensorflow::DataType dtype;
        size_t element_size;
        switch (dtype_val) {
            case 0:
                dtype = tensorflow::DT_FLOAT;
                element_size = sizeof(float);
                break;
            case 1:
                dtype = tensorflow::DT_DOUBLE;
                element_size = sizeof(double);
                break;
            case 2:
                dtype = tensorflow::DT_INT32;
                element_size = sizeof(int32_t);
                break;
            default:
                dtype = tensorflow::DT_FLOAT;
                element_size = sizeof(float);
                break;
        }
        
        size_t x_size = rows * cols * element_size;
        size_t v_size = cols * element_size;
        
        if (offset + x_size + v_size > size) return 0;
        
        // Create input tensors
        tensorflow::Tensor x_tensor(dtype, tensorflow::TensorShape({static_cast<int64_t>(rows), static_cast<int64_t>(cols)}));
        tensorflow::Tensor v_tensor(dtype, tensorflow::TensorShape({static_cast<int64_t>(cols)}));
        tensorflow::Tensor i_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        
        // Fill tensors with fuzz data
        std::memcpy(x_tensor.tensor_data().data(), data + offset, x_size);
        offset += x_size;
        std::memcpy(v_tensor.tensor_data().data(), data + offset, v_size);
        i_tensor.scalar<int32_t>()() = static_cast<int32_t>(i);
        
        // Create session and graph
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* x_node = graph_def.add_node();
        x_node->set_name("x");
        x_node->set_op("Placeholder");
        (*x_node->mutable_attr())["dtype"].set_type(dtype);
        (*x_node->mutable_attr())["shape"].mutable_shape()->add_dim()->set_size(rows);
        (*x_node->mutable_attr())["shape"].mutable_shape()->add_dim()->set_size(cols);
        
        tensorflow::NodeDef* i_node = graph_def.add_node();
        i_node->set_name("i");
        i_node->set_op("Placeholder");
        (*i_node->mutable_attr())["dtype"].set_type(tensorflow::DT_INT32);
        (*i_node->mutable_attr())["shape"].mutable_shape();
        
        tensorflow::NodeDef* v_node = graph_def.add_node();
        v_node->set_name("v");
        v_node->set_op("Placeholder");
        (*v_node->mutable_attr())["dtype"].set_type(dtype);
        (*v_node->mutable_attr())["shape"].mutable_shape()->add_dim()->set_size(cols);
        
        tensorflow::NodeDef* inplace_add_node = graph_def.add_node();
        inplace_add_node->set_name("inplace_add");
        inplace_add_node->set_op("InplaceAdd");
        inplace_add_node->add_input("x");
        inplace_add_node->add_input("i");
        inplace_add_node->add_input("v");
        (*inplace_add_node->mutable_attr())["T"].set_type(dtype);
        
        auto status = session->Create(graph_def);
        if (!status.ok()) return 0;
        
        // Run the operation
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"x", x_tensor},
            {"i", i_tensor},
            {"v", v_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"inplace_add"}, {}, &outputs);
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}