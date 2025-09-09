#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/kernels/ops_util.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 20) return 0; // Need minimum data for parameters
        
        // Extract dimensions and parameters from fuzz input
        uint32_t grad_dim0 = *reinterpret_cast<const uint32_t*>(data + offset) % 100 + 1;
        offset += 4;
        uint32_t grad_dim1 = *reinterpret_cast<const uint32_t*>(data + offset) % 100 + 1;
        offset += 4;
        uint32_t num_indices = *reinterpret_cast<const uint32_t*>(data + offset) % 50 + 1;
        offset += 4;
        uint32_t dense_output_dim0_val = *reinterpret_cast<const uint32_t*>(data + offset) % 200 + 1;
        offset += 4;
        uint32_t data_type_selector = *reinterpret_cast<const uint32_t*>(data + offset) % 4;
        offset += 4;
        
        if (offset >= size) return 0;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Select data type for grad tensor
        tensorflow::DataType grad_dtype;
        switch (data_type_selector) {
            case 0: grad_dtype = tensorflow::DT_FLOAT; break;
            case 1: grad_dtype = tensorflow::DT_DOUBLE; break;
            case 2: grad_dtype = tensorflow::DT_HALF; break;
            default: grad_dtype = tensorflow::DT_BFLOAT16; break;
        }
        
        // Create grad tensor
        tensorflow::Tensor grad_tensor(grad_dtype, tensorflow::TensorShape({static_cast<int64_t>(grad_dim0), static_cast<int64_t>(grad_dim1)}));
        
        // Fill grad tensor with data from fuzz input
        if (grad_dtype == tensorflow::DT_FLOAT) {
            auto grad_flat = grad_tensor.flat<float>();
            for (int i = 0; i < grad_flat.size() && offset < size; ++i) {
                grad_flat(i) = static_cast<float>(*reinterpret_cast<const uint8_t*>(data + offset)) / 255.0f;
                offset++;
            }
        } else if (grad_dtype == tensorflow::DT_DOUBLE) {
            auto grad_flat = grad_tensor.flat<double>();
            for (int i = 0; i < grad_flat.size() && offset < size; ++i) {
                grad_flat(i) = static_cast<double>(*reinterpret_cast<const uint8_t*>(data + offset)) / 255.0;
                offset++;
            }
        }
        
        // Create indices tensor
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({static_cast<int64_t>(num_indices)}));
        auto indices_flat = indices_tensor.flat<int32_t>();
        for (int i = 0; i < num_indices && offset < size; ++i) {
            indices_flat(i) = static_cast<int32_t>(*reinterpret_cast<const uint8_t*>(data + offset)) % grad_dim0;
            offset++;
        }
        
        // Create segment_ids tensor
        tensorflow::Tensor segment_ids_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({static_cast<int64_t>(num_indices)}));
        auto segment_ids_flat = segment_ids_tensor.flat<int32_t>();
        int32_t current_segment = 0;
        for (int i = 0; i < num_indices && offset < size; ++i) {
            if (i > 0 && (*reinterpret_cast<const uint8_t*>(data + offset) % 10) == 0) {
                current_segment++;
            }
            segment_ids_flat(i) = current_segment;
            offset++;
        }
        
        // Create dense_output_dim0 tensor
        tensorflow::Tensor dense_output_dim0_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        dense_output_dim0_tensor.scalar<int32_t>()() = static_cast<int32_t>(dense_output_dim0_val);
        
        // Create input placeholders
        auto grad_placeholder = tensorflow::ops::Placeholder(root, grad_dtype);
        auto indices_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto segment_ids_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto dense_output_dim0_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        
        // Create the SparseSegmentSumGradV2 operation
        tensorflow::Node* sparse_segment_sum_grad_v2_node;
        tensorflow::NodeBuilder builder("SparseSegmentSumGradV2", "SparseSegmentSumGradV2");
        builder.Input(grad_placeholder.node())
               .Input(indices_placeholder.node())
               .Input(segment_ids_placeholder.node())
               .Input(dense_output_dim0_placeholder.node())
               .Attr("T", grad_dtype)
               .Attr("Tidx", tensorflow::DT_INT32);
        
        tensorflow::Status status = builder.Finalize(root.graph(), &sparse_segment_sum_grad_v2_node);
        if (!status.ok()) {
            return 0;
        }
        
        // Create session and run
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        status = session->Create(root.graph()->ToGraphDef());
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {grad_placeholder.node()->name(), grad_tensor},
            {indices_placeholder.node()->name(), indices_tensor},
            {segment_ids_placeholder.node()->name(), segment_ids_tensor},
            {dense_output_dim0_placeholder.node()->name(), dense_output_dim0_tensor}
        };
        
        std::vector<std::string> output_names = {
            sparse_segment_sum_grad_v2_node->name() + ":0",  // output
            sparse_segment_sum_grad_v2_node->name() + ":1"   // sorted_unique_indices
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, output_names, {}, &outputs);
        
        if (status.ok() && outputs.size() == 2) {
            // Verify output shapes and types
            const auto& output = outputs[0];
            const auto& sorted_unique_indices = outputs[1];
            
            // Basic validation
            if (output.dtype() == grad_dtype && 
                sorted_unique_indices.dtype() == tensorflow::DT_INT32 &&
                output.dims() == 2 && 
                sorted_unique_indices.dims() == 1) {
                // Operation completed successfully
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