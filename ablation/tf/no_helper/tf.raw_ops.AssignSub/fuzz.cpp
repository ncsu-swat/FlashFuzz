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
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/ops/state_ops.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract parameters from fuzzer input
        uint32_t dtype_val = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        
        uint32_t shape_size = *reinterpret_cast<const uint32_t*>(data + offset) % 4 + 1;
        offset += 4;
        
        bool use_locking = (*reinterpret_cast<const uint8_t*>(data + offset)) % 2;
        offset += 1;
        
        if (offset + shape_size * 4 + 8 > size) return 0;
        
        // Map to valid TensorFlow data types
        tensorflow::DataType dtype;
        switch (dtype_val % 8) {
            case 0: dtype = tensorflow::DT_FLOAT; break;
            case 1: dtype = tensorflow::DT_DOUBLE; break;
            case 2: dtype = tensorflow::DT_INT32; break;
            case 3: dtype = tensorflow::DT_INT64; break;
            case 4: dtype = tensorflow::DT_UINT8; break;
            case 5: dtype = tensorflow::DT_INT16; break;
            case 6: dtype = tensorflow::DT_INT8; break;
            default: dtype = tensorflow::DT_FLOAT; break;
        }
        
        // Create tensor shape
        tensorflow::TensorShape shape;
        for (uint32_t i = 0; i < shape_size && offset + 4 <= size; ++i) {
            int32_t dim = *reinterpret_cast<const int32_t*>(data + offset);
            offset += 4;
            dim = std::abs(dim) % 10 + 1; // Keep dimensions reasonable
            shape.AddDim(dim);
        }
        
        if (shape.num_elements() > 1000) return 0; // Limit tensor size
        
        // Create scope and session
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create variable
        auto var = tensorflow::ops::Variable(root.WithOpName("var"), shape, dtype);
        
        // Initialize variable with some values
        tensorflow::Tensor init_tensor(dtype, shape);
        
        // Fill tensor with data from fuzzer input
        size_t tensor_bytes = init_tensor.TotalBytes();
        if (offset + tensor_bytes > size) {
            // Fill with pattern if not enough data
            auto flat = init_tensor.flat<float>();
            for (int i = 0; i < flat.size(); ++i) {
                flat(i) = (i % 256) / 256.0f;
            }
        } else {
            std::memcpy(init_tensor.tensor_data().data(), data + offset, 
                       std::min(tensor_bytes, size - offset));
            offset += tensor_bytes;
        }
        
        auto assign_init = tensorflow::ops::Assign(root.WithOpName("assign_init"), var, init_tensor);
        
        // Create value tensor for subtraction
        tensorflow::Tensor value_tensor(dtype, shape);
        if (offset + tensor_bytes <= size) {
            std::memcpy(value_tensor.tensor_data().data(), data + offset, 
                       std::min(tensor_bytes, size - offset));
        } else {
            // Fill with pattern
            auto flat = value_tensor.flat<float>();
            for (int i = 0; i < flat.size(); ++i) {
                flat(i) = (i % 128) / 128.0f;
            }
        }
        
        // Create AssignSub operation
        auto assign_sub = tensorflow::ops::AssignSub(root.WithOpName("assign_sub"), 
                                                    var, value_tensor,
                                                    tensorflow::ops::AssignSub::UseLocking(use_locking));
        
        // Create session and run
        tensorflow::ClientSession session(root);
        
        // Initialize variable
        std::vector<tensorflow::Tensor> init_outputs;
        auto init_status = session.Run({assign_init}, &init_outputs);
        if (!init_status.ok()) {
            return 0;
        }
        
        // Run AssignSub
        std::vector<tensorflow::Tensor> outputs;
        auto status = session.Run({assign_sub}, &outputs);
        if (!status.ok()) {
            return 0;
        }
        
        // Verify output
        if (!outputs.empty()) {
            const auto& result = outputs[0];
            if (result.dtype() != dtype || result.shape() != shape) {
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