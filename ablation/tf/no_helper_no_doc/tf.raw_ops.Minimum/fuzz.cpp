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
        uint32_t dim1 = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        uint32_t dim2 = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        uint32_t dtype_val = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        uint32_t num_elements = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        
        // Limit dimensions to reasonable values
        dim1 = (dim1 % 100) + 1;
        dim2 = (dim2 % 100) + 1;
        num_elements = std::min(num_elements % 1000 + 1, dim1 * dim2);
        
        // Select data type
        tensorflow::DataType dtype;
        switch (dtype_val % 4) {
            case 0: dtype = tensorflow::DT_FLOAT; break;
            case 1: dtype = tensorflow::DT_DOUBLE; break;
            case 2: dtype = tensorflow::DT_INT32; break;
            case 3: dtype = tensorflow::DT_INT64; break;
            default: dtype = tensorflow::DT_FLOAT; break;
        }
        
        // Create tensor shapes
        tensorflow::TensorShape shape({static_cast<int64_t>(dim1), static_cast<int64_t>(dim2)});
        
        // Create input tensors
        tensorflow::Tensor x_tensor(dtype, shape);
        tensorflow::Tensor y_tensor(dtype, shape);
        
        // Fill tensors with fuzz data
        size_t element_size = 0;
        switch (dtype) {
            case tensorflow::DT_FLOAT:
                element_size = sizeof(float);
                break;
            case tensorflow::DT_DOUBLE:
                element_size = sizeof(double);
                break;
            case tensorflow::DT_INT32:
                element_size = sizeof(int32_t);
                break;
            case tensorflow::DT_INT64:
                element_size = sizeof(int64_t);
                break;
        }
        
        size_t total_bytes_needed = num_elements * element_size * 2;
        if (offset + total_bytes_needed > size) {
            // Not enough data, fill with zeros
            memset(x_tensor.data(), 0, x_tensor.TotalBytes());
            memset(y_tensor.data(), 0, y_tensor.TotalBytes());
        } else {
            // Copy available data
            size_t copy_size = std::min(x_tensor.TotalBytes(), size - offset);
            memcpy(x_tensor.data(), data + offset, copy_size);
            offset += copy_size;
            
            copy_size = std::min(y_tensor.TotalBytes(), size - offset);
            if (offset < size) {
                memcpy(y_tensor.data(), data + offset, copy_size);
            } else {
                memset(y_tensor.data(), 0, y_tensor.TotalBytes());
            }
        }
        
        // Create session and graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* x_node = graph_def.add_node();
        x_node->set_name("x");
        x_node->set_op("Placeholder");
        (*x_node->mutable_attr())["dtype"].set_type(dtype);
        
        tensorflow::NodeDef* y_node = graph_def.add_node();
        y_node->set_name("y");
        y_node->set_op("Placeholder");
        (*y_node->mutable_attr())["dtype"].set_type(dtype);
        
        tensorflow::NodeDef* min_node = graph_def.add_node();
        min_node->set_name("minimum");
        min_node->set_op("Minimum");
        min_node->add_input("x");
        min_node->add_input("y");
        (*min_node->mutable_attr())["T"].set_type(dtype);
        
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        if (!session) return 0;
        
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) return 0;
        
        // Run the operation
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"x", x_tensor},
            {"y", y_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"minimum"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Operation succeeded, verify output shape matches input
            if (outputs[0].shape() == shape && outputs[0].dtype() == dtype) {
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