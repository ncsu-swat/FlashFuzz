#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/kernels/scatter_functor.h>
#include <tensorflow/core/platform/types.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions and parameters from fuzz input
        int32_t ref_dim0 = *reinterpret_cast<const int32_t*>(data + offset) % 100 + 1;
        offset += 4;
        int32_t ref_dim1 = *reinterpret_cast<const int32_t*>(data + offset) % 100 + 1;
        offset += 4;
        int32_t indices_size = *reinterpret_cast<const int32_t*>(data + offset) % 50 + 1;
        offset += 4;
        int32_t updates_dim1 = ref_dim1;
        offset += 4;
        
        if (offset >= size) return 0;
        
        // Create TensorFlow scope
        auto root = tensorflow::Scope::NewRootScope();
        
        // Create ref tensor (variable to be updated)
        tensorflow::TensorShape ref_shape({ref_dim0, ref_dim1});
        tensorflow::Tensor ref_tensor(tensorflow::DT_FLOAT, ref_shape);
        auto ref_flat = ref_tensor.flat<float>();
        
        // Fill ref tensor with data from fuzz input
        for (int i = 0; i < ref_flat.size() && offset + 4 <= size; ++i) {
            ref_flat(i) = *reinterpret_cast<const float*>(data + offset);
            offset += 4;
            if (offset >= size) break;
        }
        
        // Create indices tensor
        tensorflow::TensorShape indices_shape({indices_size});
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, indices_shape);
        auto indices_flat = indices_tensor.flat<int32_t>();
        
        // Fill indices tensor with valid indices
        for (int i = 0; i < indices_size && offset + 4 <= size; ++i) {
            int32_t idx = *reinterpret_cast<const int32_t*>(data + offset);
            indices_flat(i) = std::abs(idx) % ref_dim0;  // Ensure valid index
            offset += 4;
            if (offset >= size) break;
        }
        
        // Create updates tensor
        tensorflow::TensorShape updates_shape({indices_size, updates_dim1});
        tensorflow::Tensor updates_tensor(tensorflow::DT_FLOAT, updates_shape);
        auto updates_flat = updates_tensor.flat<float>();
        
        // Fill updates tensor with data from fuzz input
        for (int i = 0; i < updates_flat.size() && offset + 4 <= size; ++i) {
            updates_flat(i) = *reinterpret_cast<const float*>(data + offset);
            offset += 4;
            if (offset >= size) break;
        }
        
        // Create Variable node for ref
        auto var = tensorflow::ops::Variable(root, ref_shape, tensorflow::DT_FLOAT);
        
        // Create Assign node to initialize the variable
        auto assign = tensorflow::ops::Assign(root, var, tensorflow::ops::Const(root, ref_tensor));
        
        // Create ScatterMax operation
        auto scatter_max = tensorflow::ops::ScatterMax(
            root, 
            var,
            tensorflow::ops::Const(root, indices_tensor),
            tensorflow::ops::Const(root, updates_tensor)
        );
        
        // Create session and run the operations
        tensorflow::ClientSession session(root);
        
        // Initialize the variable
        std::vector<tensorflow::Tensor> init_outputs;
        auto init_status = session.Run({assign}, &init_outputs);
        if (!init_status.ok()) {
            return 0;  // Silently handle initialization errors
        }
        
        // Run ScatterMax operation
        std::vector<tensorflow::Tensor> outputs;
        auto status = session.Run({scatter_max}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Verify output tensor properties
            const auto& output = outputs[0];
            if (output.dtype() == tensorflow::DT_FLOAT && 
                output.shape().dims() == ref_shape.dims()) {
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