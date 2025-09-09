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
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/const_op.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 20) return 0;
        
        // Extract dimensions and parameters from fuzz input
        uint32_t ref_dim0 = *reinterpret_cast<const uint32_t*>(data + offset) % 100 + 1;
        offset += 4;
        uint32_t ref_dim1 = *reinterpret_cast<const uint32_t*>(data + offset) % 100 + 1;
        offset += 4;
        uint32_t indices_size = *reinterpret_cast<const uint32_t*>(data + offset) % 50 + 1;
        offset += 4;
        uint32_t data_type = *reinterpret_cast<const uint32_t*>(data + offset) % 4;
        offset += 4;
        bool use_locking = (*reinterpret_cast<const uint32_t*>(data + offset) % 2) == 1;
        offset += 4;
        
        tensorflow::DataType dtype;
        switch (data_type) {
            case 0: dtype = tensorflow::DT_FLOAT; break;
            case 1: dtype = tensorflow::DT_DOUBLE; break;
            case 2: dtype = tensorflow::DT_INT32; break;
            case 3: dtype = tensorflow::DT_INT64; break;
            default: dtype = tensorflow::DT_FLOAT; break;
        }
        
        // Create TensorFlow scope
        auto root = tensorflow::Scope::NewRootScope();
        
        // Create ref tensor (variable)
        tensorflow::TensorShape ref_shape({static_cast<int64_t>(ref_dim0), static_cast<int64_t>(ref_dim1)});
        tensorflow::Tensor ref_tensor(dtype, ref_shape);
        
        // Initialize ref tensor with data from fuzz input
        size_t ref_elements = ref_dim0 * ref_dim1;
        size_t bytes_per_element = 4; // Assume 4 bytes per element for simplicity
        
        if (offset + ref_elements * bytes_per_element > size) {
            // Not enough data, use what we have
            ref_elements = (size - offset) / bytes_per_element;
        }
        
        if (dtype == tensorflow::DT_FLOAT) {
            auto ref_flat = ref_tensor.flat<float>();
            for (size_t i = 0; i < ref_elements && offset + 4 <= size; ++i) {
                float val = *reinterpret_cast<const float*>(data + offset);
                ref_flat(i) = val;
                offset += 4;
            }
        } else if (dtype == tensorflow::DT_INT32) {
            auto ref_flat = ref_tensor.flat<int32_t>();
            for (size_t i = 0; i < ref_elements && offset + 4 <= size; ++i) {
                int32_t val = *reinterpret_cast<const int32_t*>(data + offset);
                ref_flat(i) = val;
                offset += 4;
            }
        }
        
        // Create indices tensor
        tensorflow::TensorShape indices_shape({static_cast<int64_t>(indices_size)});
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, indices_shape);
        auto indices_flat = indices_tensor.flat<int32_t>();
        
        for (size_t i = 0; i < indices_size && offset + 4 <= size; ++i) {
            int32_t idx = *reinterpret_cast<const int32_t*>(data + offset) % ref_dim0;
            indices_flat(i) = idx;
            offset += 4;
        }
        
        // Create updates tensor
        tensorflow::TensorShape updates_shape({static_cast<int64_t>(indices_size), static_cast<int64_t>(ref_dim1)});
        tensorflow::Tensor updates_tensor(dtype, updates_shape);
        
        size_t updates_elements = indices_size * ref_dim1;
        if (dtype == tensorflow::DT_FLOAT) {
            auto updates_flat = updates_tensor.flat<float>();
            for (size_t i = 0; i < updates_elements && offset + 4 <= size; ++i) {
                float val = *reinterpret_cast<const float*>(data + offset);
                updates_flat(i) = val;
                offset += 4;
            }
        } else if (dtype == tensorflow::DT_INT32) {
            auto updates_flat = updates_tensor.flat<int32_t>();
            for (size_t i = 0; i < updates_elements && offset + 4 <= size; ++i) {
                int32_t val = *reinterpret_cast<const int32_t*>(data + offset);
                updates_flat(i) = val;
                offset += 4;
            }
        }
        
        // Create Variable node for ref
        auto ref_var = tensorflow::ops::Variable(root, ref_shape, dtype);
        auto assign_ref = tensorflow::ops::Assign(root, ref_var, tensorflow::ops::Const(root, ref_tensor));
        
        // Create ScatterMax operation
        auto scatter_max = tensorflow::ops::ScatterMax(
            root,
            ref_var,
            tensorflow::ops::Const(root, indices_tensor),
            tensorflow::ops::Const(root, updates_tensor),
            tensorflow::ops::ScatterMax::UseLocking(use_locking)
        );
        
        // Create session and run
        tensorflow::ClientSession session(root);
        
        // Initialize variable
        std::vector<tensorflow::Tensor> init_outputs;
        auto init_status = session.Run({assign_ref}, &init_outputs);
        if (!init_status.ok()) {
            return 0; // Ignore errors in fuzzing
        }
        
        // Run ScatterMax
        std::vector<tensorflow::Tensor> outputs;
        auto status = session.Run({scatter_max}, &outputs);
        if (!status.ok()) {
            return 0; // Ignore errors in fuzzing
        }
        
        // Verify output tensor has correct shape and type
        if (!outputs.empty()) {
            const auto& output = outputs[0];
            if (output.shape().dims() == 2 && 
                output.shape().dim_size(0) == ref_dim0 &&
                output.shape().dim_size(1) == ref_dim1 &&
                output.dtype() == dtype) {
                // Basic validation passed
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}