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
#include <tensorflow/core/graph/graph_def_builder.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions and data type
        uint32_t x_dim1 = *reinterpret_cast<const uint32_t*>(data + offset) % 10 + 1;
        offset += 4;
        uint32_t x_dim2 = *reinterpret_cast<const uint32_t*>(data + offset) % 10 + 1;
        offset += 4;
        uint32_t y_dim1 = *reinterpret_cast<const uint32_t*>(data + offset) % 10 + 1;
        offset += 4;
        uint32_t y_dim2 = *reinterpret_cast<const uint32_t*>(data + offset) % 10 + 1;
        offset += 4;
        
        if (offset >= size) return 0;
        
        // Choose data type based on fuzzer input
        tensorflow::DataType dtype;
        uint8_t type_selector = data[offset] % 8;
        offset++;
        
        switch (type_selector) {
            case 0: dtype = tensorflow::DT_FLOAT; break;
            case 1: dtype = tensorflow::DT_DOUBLE; break;
            case 2: dtype = tensorflow::DT_INT32; break;
            case 3: dtype = tensorflow::DT_INT64; break;
            case 4: dtype = tensorflow::DT_INT8; break;
            case 5: dtype = tensorflow::DT_UINT8; break;
            case 6: dtype = tensorflow::DT_INT16; break;
            case 7: dtype = tensorflow::DT_UINT16; break;
            default: dtype = tensorflow::DT_FLOAT; break;
        }
        
        // Create tensor shapes
        tensorflow::TensorShape x_shape({x_dim1, x_dim2});
        tensorflow::TensorShape y_shape({y_dim1, y_dim2});
        
        // Create tensors
        tensorflow::Tensor x_tensor(dtype, x_shape);
        tensorflow::Tensor y_tensor(dtype, y_shape);
        
        size_t x_elements = x_dim1 * x_dim2;
        size_t y_elements = y_dim1 * y_dim2;
        
        // Fill tensors with fuzzer data
        if (dtype == tensorflow::DT_FLOAT) {
            auto x_flat = x_tensor.flat<float>();
            auto y_flat = y_tensor.flat<float>();
            
            for (size_t i = 0; i < x_elements && offset + 4 <= size; ++i) {
                float val = *reinterpret_cast<const float*>(data + offset);
                x_flat(i) = val;
                offset += 4;
            }
            
            for (size_t i = 0; i < y_elements && offset + 4 <= size; ++i) {
                float val = *reinterpret_cast<const float*>(data + offset);
                y_flat(i) = val;
                offset += 4;
            }
        } else if (dtype == tensorflow::DT_INT32) {
            auto x_flat = x_tensor.flat<int32_t>();
            auto y_flat = y_tensor.flat<int32_t>();
            
            for (size_t i = 0; i < x_elements && offset + 4 <= size; ++i) {
                int32_t val = *reinterpret_cast<const int32_t*>(data + offset);
                x_flat(i) = val;
                offset += 4;
            }
            
            for (size_t i = 0; i < y_elements && offset + 4 <= size; ++i) {
                int32_t val = *reinterpret_cast<const int32_t*>(data + offset);
                y_flat(i) = val;
                offset += 4;
            }
        } else if (dtype == tensorflow::DT_DOUBLE) {
            auto x_flat = x_tensor.flat<double>();
            auto y_flat = y_tensor.flat<double>();
            
            for (size_t i = 0; i < x_elements && offset + 8 <= size; ++i) {
                double val = *reinterpret_cast<const double*>(data + offset);
                x_flat(i) = val;
                offset += 8;
            }
            
            for (size_t i = 0; i < y_elements && offset + 8 <= size; ++i) {
                double val = *reinterpret_cast<const double*>(data + offset);
                y_flat(i) = val;
                offset += 8;
            }
        } else {
            // For other types, fill with simple pattern
            if (dtype == tensorflow::DT_INT8) {
                auto x_flat = x_tensor.flat<int8_t>();
                auto y_flat = y_tensor.flat<int8_t>();
                for (size_t i = 0; i < x_elements && offset < size; ++i) {
                    x_flat(i) = static_cast<int8_t>(data[offset++]);
                }
                for (size_t i = 0; i < y_elements && offset < size; ++i) {
                    y_flat(i) = static_cast<int8_t>(data[offset++]);
                }
            }
        }
        
        // Create a simple session and graph
        tensorflow::GraphDefBuilder builder(tensorflow::GraphDefBuilder::kFailImmediately);
        
        tensorflow::Node* x_node = tensorflow::ops::Const(x_tensor, builder.opts().WithName("x"));
        tensorflow::Node* y_node = tensorflow::ops::Const(y_tensor, builder.opts().WithName("y"));
        
        tensorflow::Node* minimum_node = tensorflow::ops::BinaryOp("Minimum", x_node, y_node, 
                                                                   builder.opts().WithName("minimum"));
        
        tensorflow::GraphDef graph_def;
        tensorflow::Status status = builder.ToGraphDef(&graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Create session and run
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({}, {"minimum:0"}, {}, &outputs);
        if (!status.ok()) {
            return 0;
        }
        
        // Verify output has expected properties
        if (!outputs.empty()) {
            const tensorflow::Tensor& result = outputs[0];
            if (result.dtype() != dtype) {
                return 0;
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}