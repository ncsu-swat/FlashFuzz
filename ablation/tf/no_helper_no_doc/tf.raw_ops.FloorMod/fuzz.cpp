#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/framework/node_def_util.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions and data type info
        uint32_t dim1 = *reinterpret_cast<const uint32_t*>(data + offset) % 10 + 1;
        offset += 4;
        uint32_t dim2 = *reinterpret_cast<const uint32_t*>(data + offset) % 10 + 1;
        offset += 4;
        uint32_t data_type = *reinterpret_cast<const uint32_t*>(data + offset) % 3;
        offset += 4;
        
        tensorflow::DataType dtype;
        size_t element_size;
        switch (data_type) {
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
        
        size_t total_elements = dim1 * dim2;
        size_t required_size = total_elements * element_size * 2; // for both tensors
        
        if (offset + required_size > size) return 0;
        
        // Create tensor shapes
        tensorflow::TensorShape shape({static_cast<int64_t>(dim1), static_cast<int64_t>(dim2)});
        
        // Create input tensors
        tensorflow::Tensor x_tensor(dtype, shape);
        tensorflow::Tensor y_tensor(dtype, shape);
        
        // Fill tensors with fuzz data
        if (dtype == tensorflow::DT_FLOAT) {
            auto x_flat = x_tensor.flat<float>();
            auto y_flat = y_tensor.flat<float>();
            
            for (size_t i = 0; i < total_elements; ++i) {
                if (offset + sizeof(float) <= size) {
                    x_flat(i) = *reinterpret_cast<const float*>(data + offset);
                    offset += sizeof(float);
                } else {
                    x_flat(i) = 1.0f;
                }
            }
            
            for (size_t i = 0; i < total_elements; ++i) {
                if (offset + sizeof(float) <= size) {
                    float val = *reinterpret_cast<const float*>(data + offset);
                    y_flat(i) = (val == 0.0f) ? 1.0f : val; // Avoid division by zero
                    offset += sizeof(float);
                } else {
                    y_flat(i) = 1.0f;
                }
            }
        } else if (dtype == tensorflow::DT_DOUBLE) {
            auto x_flat = x_tensor.flat<double>();
            auto y_flat = y_tensor.flat<double>();
            
            for (size_t i = 0; i < total_elements; ++i) {
                if (offset + sizeof(double) <= size) {
                    x_flat(i) = *reinterpret_cast<const double*>(data + offset);
                    offset += sizeof(double);
                } else {
                    x_flat(i) = 1.0;
                }
            }
            
            for (size_t i = 0; i < total_elements; ++i) {
                if (offset + sizeof(double) <= size) {
                    double val = *reinterpret_cast<const double*>(data + offset);
                    y_flat(i) = (val == 0.0) ? 1.0 : val; // Avoid division by zero
                    offset += sizeof(double);
                } else {
                    y_flat(i) = 1.0;
                }
            }
        } else if (dtype == tensorflow::DT_INT32) {
            auto x_flat = x_tensor.flat<int32_t>();
            auto y_flat = y_tensor.flat<int32_t>();
            
            for (size_t i = 0; i < total_elements; ++i) {
                if (offset + sizeof(int32_t) <= size) {
                    x_flat(i) = *reinterpret_cast<const int32_t*>(data + offset);
                    offset += sizeof(int32_t);
                } else {
                    x_flat(i) = 1;
                }
            }
            
            for (size_t i = 0; i < total_elements; ++i) {
                if (offset + sizeof(int32_t) <= size) {
                    int32_t val = *reinterpret_cast<const int32_t*>(data + offset);
                    y_flat(i) = (val == 0) ? 1 : val; // Avoid division by zero
                    offset += sizeof(int32_t);
                } else {
                    y_flat(i) = 1;
                }
            }
        }
        
        // Create a simple computation graph using TensorFlow C++ API
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto x_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto y_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto floor_mod = tensorflow::ops::FloorMod(root, x_placeholder, y_placeholder);
        
        tensorflow::GraphDef graph;
        TF_CHECK_OK(root.ToGraphDef(&graph));
        
        // Create session and run the operation
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        TF_CHECK_OK(session->Create(graph));
        
        std::vector<tensorflow::Tensor> outputs;
        TF_CHECK_OK(session->Run({{x_placeholder.node()->name(), x_tensor},
                                  {y_placeholder.node()->name(), y_tensor}},
                                 {floor_mod.node()->name()}, {}, &outputs));
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}