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
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/const_op.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions
        uint32_t batch_size = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        uint32_t num_classes = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        
        // Limit dimensions to reasonable values
        batch_size = (batch_size % 100) + 1;
        num_classes = (num_classes % 100) + 1;
        
        // Extract data type
        uint32_t dtype_val = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        
        tensorflow::DataType dtype;
        switch (dtype_val % 4) {
            case 0: dtype = tensorflow::DT_HALF; break;
            case 1: dtype = tensorflow::DT_BFLOAT16; break;
            case 2: dtype = tensorflow::DT_FLOAT; break;
            case 3: dtype = tensorflow::DT_DOUBLE; break;
            default: dtype = tensorflow::DT_FLOAT; break;
        }
        
        // Calculate required data size
        size_t element_count = batch_size * num_classes;
        size_t element_size = 4; // Default to float32 size
        
        if (dtype == tensorflow::DT_HALF || dtype == tensorflow::DT_BFLOAT16) {
            element_size = 2;
        } else if (dtype == tensorflow::DT_DOUBLE) {
            element_size = 8;
        }
        
        size_t required_data_size = element_count * element_size;
        
        if (offset + required_data_size > size) return 0;
        
        // Create TensorFlow scope and session
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input tensor shape
        tensorflow::TensorShape shape({static_cast<int64_t>(batch_size), static_cast<int64_t>(num_classes)});
        
        // Create input tensor
        tensorflow::Tensor input_tensor(dtype, shape);
        
        // Fill tensor with fuzz data
        if (dtype == tensorflow::DT_FLOAT) {
            auto tensor_data = input_tensor.flat<float>();
            const float* fuzz_data = reinterpret_cast<const float*>(data + offset);
            for (size_t i = 0; i < element_count && i < tensor_data.size(); ++i) {
                tensor_data(i) = fuzz_data[i];
            }
        } else if (dtype == tensorflow::DT_DOUBLE) {
            auto tensor_data = input_tensor.flat<double>();
            const double* fuzz_data = reinterpret_cast<const double*>(data + offset);
            for (size_t i = 0; i < element_count && i < tensor_data.size(); ++i) {
                tensor_data(i) = fuzz_data[i];
            }
        } else if (dtype == tensorflow::DT_HALF) {
            auto tensor_data = input_tensor.flat<Eigen::half>();
            const uint16_t* fuzz_data = reinterpret_cast<const uint16_t*>(data + offset);
            for (size_t i = 0; i < element_count && i < tensor_data.size(); ++i) {
                tensor_data(i) = Eigen::half_impl::raw_uint16_to_half(fuzz_data[i]);
            }
        } else if (dtype == tensorflow::DT_BFLOAT16) {
            auto tensor_data = input_tensor.flat<tensorflow::bfloat16>();
            const uint16_t* fuzz_data = reinterpret_cast<const uint16_t*>(data + offset);
            for (size_t i = 0; i < element_count && i < tensor_data.size(); ++i) {
                tensor_data(i) = tensorflow::bfloat16(fuzz_data[i]);
            }
        }
        
        // Create placeholder for input
        auto logits = tensorflow::ops::Placeholder(root, dtype);
        
        // Create Softmax operation
        auto softmax = tensorflow::ops::Softmax(root, logits);
        
        // Create session
        tensorflow::ClientSession session(root);
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({{logits, input_tensor}}, {softmax}, &outputs);
        
        if (!status.ok()) {
            std::cout << "TensorFlow operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output
        if (!outputs.empty()) {
            const tensorflow::Tensor& output = outputs[0];
            if (output.shape() != shape) {
                std::cout << "Output shape mismatch" << std::endl;
                return 0;
            }
            
            if (output.dtype() != dtype) {
                std::cout << "Output dtype mismatch" << std::endl;
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