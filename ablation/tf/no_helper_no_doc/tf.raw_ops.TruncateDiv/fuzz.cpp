#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/common_runtime/kernel_benchmark_testlib.h>
#include <tensorflow/core/framework/fake_input.h>
#include <tensorflow/core/framework/node_def_builder.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/types.pb.h>
#include <tensorflow/core/kernels/ops_testutil.h>
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions and data type info
        uint32_t dim1 = *reinterpret_cast<const uint32_t*>(data + offset) % 10 + 1;
        offset += 4;
        uint32_t dim2 = *reinterpret_cast<const uint32_t*>(data + offset) % 10 + 1;
        offset += 4;
        uint32_t type_selector = *reinterpret_cast<const uint32_t*>(data + offset) % 4;
        offset += 4;
        
        tensorflow::DataType dtype;
        size_t element_size;
        switch (type_selector) {
            case 0:
                dtype = tensorflow::DT_INT32;
                element_size = sizeof(int32_t);
                break;
            case 1:
                dtype = tensorflow::DT_INT64;
                element_size = sizeof(int64_t);
                break;
            case 2:
                dtype = tensorflow::DT_FLOAT;
                element_size = sizeof(float);
                break;
            case 3:
                dtype = tensorflow::DT_DOUBLE;
                element_size = sizeof(double);
                break;
            default:
                dtype = tensorflow::DT_INT32;
                element_size = sizeof(int32_t);
                break;
        }
        
        size_t total_elements = dim1 * dim2;
        size_t required_size = offset + 2 * total_elements * element_size;
        
        if (size < required_size) return 0;
        
        // Create tensor shapes
        tensorflow::TensorShape shape({static_cast<int64_t>(dim1), static_cast<int64_t>(dim2)});
        
        // Create input tensors
        tensorflow::Tensor x_tensor(dtype, shape);
        tensorflow::Tensor y_tensor(dtype, shape);
        
        // Fill tensors with fuzz data
        const uint8_t* x_data = data + offset;
        const uint8_t* y_data = data + offset + total_elements * element_size;
        
        if (dtype == tensorflow::DT_INT32) {
            auto x_flat = x_tensor.flat<int32_t>();
            auto y_flat = y_tensor.flat<int32_t>();
            
            for (size_t i = 0; i < total_elements; ++i) {
                int32_t x_val = *reinterpret_cast<const int32_t*>(x_data + i * element_size);
                int32_t y_val = *reinterpret_cast<const int32_t*>(y_data + i * element_size);
                
                // Avoid division by zero
                if (y_val == 0) y_val = 1;
                
                x_flat(i) = x_val;
                y_flat(i) = y_val;
            }
        } else if (dtype == tensorflow::DT_INT64) {
            auto x_flat = x_tensor.flat<int64_t>();
            auto y_flat = y_tensor.flat<int64_t>();
            
            for (size_t i = 0; i < total_elements; ++i) {
                int64_t x_val = *reinterpret_cast<const int64_t*>(x_data + i * element_size);
                int64_t y_val = *reinterpret_cast<const int64_t*>(y_data + i * element_size);
                
                // Avoid division by zero
                if (y_val == 0) y_val = 1;
                
                x_flat(i) = x_val;
                y_flat(i) = y_val;
            }
        } else if (dtype == tensorflow::DT_FLOAT) {
            auto x_flat = x_tensor.flat<float>();
            auto y_flat = y_tensor.flat<float>();
            
            for (size_t i = 0; i < total_elements; ++i) {
                float x_val = *reinterpret_cast<const float*>(x_data + i * element_size);
                float y_val = *reinterpret_cast<const float*>(y_data + i * element_size);
                
                // Avoid division by zero and NaN/inf
                if (y_val == 0.0f || !std::isfinite(y_val)) y_val = 1.0f;
                if (!std::isfinite(x_val)) x_val = 1.0f;
                
                x_flat(i) = x_val;
                y_flat(i) = y_val;
            }
        } else if (dtype == tensorflow::DT_DOUBLE) {
            auto x_flat = x_tensor.flat<double>();
            auto y_flat = y_tensor.flat<double>();
            
            for (size_t i = 0; i < total_elements; ++i) {
                double x_val = *reinterpret_cast<const double*>(x_data + i * element_size);
                double y_val = *reinterpret_cast<const double*>(y_data + i * element_size);
                
                // Avoid division by zero and NaN/inf
                if (y_val == 0.0 || !std::isfinite(y_val)) y_val = 1.0;
                if (!std::isfinite(x_val)) x_val = 1.0;
                
                x_flat(i) = x_val;
                y_flat(i) = y_val;
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
        
        tensorflow::NodeDef* truncate_div_node = graph_def.add_node();
        truncate_div_node->set_name("truncate_div");
        truncate_div_node->set_op("TruncateDiv");
        truncate_div_node->add_input("x");
        truncate_div_node->add_input("y");
        (*truncate_div_node->mutable_attr())["T"].set_type(dtype);
        
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run the operation
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"x", x_tensor},
            {"y", y_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"truncate_div"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Operation completed successfully
        }
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}