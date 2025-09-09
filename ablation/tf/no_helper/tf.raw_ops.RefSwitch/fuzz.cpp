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
        
        if (size < 16) return 0;
        
        // Extract tensor dimensions and pred value from fuzz input
        int32_t dim1 = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        int32_t dim2 = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        bool pred_value = *reinterpret_cast<const bool*>(data + offset);
        offset += sizeof(bool);
        
        // Clamp dimensions to reasonable values
        dim1 = std::max(1, std::min(dim1, 100));
        dim2 = std::max(1, std::min(dim2, 100));
        
        // Calculate remaining data size for tensor values
        size_t remaining_size = size - offset;
        size_t tensor_elements = dim1 * dim2;
        size_t bytes_per_element = sizeof(float);
        
        if (remaining_size < tensor_elements * bytes_per_element) {
            return 0;
        }
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input tensor data
        tensorflow::TensorShape data_shape({dim1, dim2});
        tensorflow::Tensor data_tensor(tensorflow::DT_FLOAT, data_shape);
        auto data_flat = data_tensor.flat<float>();
        
        // Fill tensor with fuzz data
        for (int i = 0; i < tensor_elements && offset + bytes_per_element <= size; ++i) {
            float value = *reinterpret_cast<const float*>(data + offset);
            // Handle NaN and infinity values
            if (std::isnan(value) || std::isinf(value)) {
                value = 0.0f;
            }
            data_flat(i) = value;
            offset += bytes_per_element;
        }
        
        // Create predicate tensor
        tensorflow::Tensor pred_tensor(tensorflow::DT_BOOL, tensorflow::TensorShape({}));
        pred_tensor.scalar<bool>()() = pred_value;
        
        // Create Variable for mutable tensor (RefSwitch requires ref tensor)
        auto var_data = tensorflow::ops::Variable(root.WithOpName("var_data"), 
                                                 data_shape, tensorflow::DT_FLOAT);
        
        // Assign initial value to variable
        auto assign_op = tensorflow::ops::Assign(root.WithOpName("assign"), 
                                               var_data, 
                                               tensorflow::ops::Const(root, data_tensor));
        
        // Create predicate constant
        auto pred_const = tensorflow::ops::Const(root.WithOpName("pred"), pred_tensor);
        
        // Create RefSwitch operation
        auto ref_switch = tensorflow::ops::RefSwitch(root.WithOpName("ref_switch"), 
                                                   var_data, 
                                                   pred_const);
        
        // Create session and run
        tensorflow::ClientSession session(root);
        
        // Initialize variable
        std::vector<tensorflow::Tensor> assign_outputs;
        tensorflow::Status assign_status = session.Run({assign_op}, &assign_outputs);
        if (!assign_status.ok()) {
            return 0;
        }
        
        // Run RefSwitch
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({ref_switch.output_false, ref_switch.output_true}, &outputs);
        
        if (status.ok() && outputs.size() == 2) {
            // Verify output shapes match input
            if (outputs[0].shape() == data_shape && outputs[1].shape() == data_shape) {
                // Basic validation - one output should have data, other should be uninitialized or empty
                // depending on pred_value
                auto output_false_flat = outputs[0].flat<float>();
                auto output_true_flat = outputs[1].flat<float>();
                
                // Simple validation that outputs exist
                if (output_false_flat.size() == tensor_elements && 
                    output_true_flat.size() == tensor_elements) {
                    // Test passed
                }
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}