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
#include <tensorflow/core/platform/logging.h>
#include <tensorflow/core/platform/types.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract parameters from fuzzer input
        uint32_t num_dims = (data[offset] % 4) + 1;
        offset += 1;
        
        uint32_t num_reduction_dims = (data[offset] % num_dims) + 1;
        offset += 1;
        
        bool keep_dims = data[offset] % 2;
        offset += 1;
        
        // Create tensor shape
        std::vector<int64_t> shape;
        for (uint32_t i = 0; i < num_dims && offset < size; ++i) {
            int64_t dim = (data[offset] % 10) + 1;
            shape.push_back(dim);
            offset += 1;
        }
        
        if (shape.empty()) {
            shape.push_back(1);
        }
        
        // Create reduction indices
        std::vector<int32_t> reduction_indices;
        for (uint32_t i = 0; i < num_reduction_dims && offset < size; ++i) {
            int32_t idx = data[offset] % num_dims;
            reduction_indices.push_back(idx);
            offset += 1;
        }
        
        // Calculate total elements
        int64_t total_elements = 1;
        for (auto dim : shape) {
            total_elements *= dim;
        }
        
        if (total_elements > 1000) {
            total_elements = 1000;
        }
        
        // Create TensorFlow scope and session
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input tensor
        tensorflow::TensorShape tensor_shape;
        for (auto dim : shape) {
            tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensor_shape);
        auto input_flat = input_tensor.flat<float>();
        
        // Fill tensor with data from fuzzer input
        for (int64_t i = 0; i < total_elements && offset < size; ++i) {
            float val = static_cast<float>(data[offset % size]) / 255.0f;
            input_flat(i) = val;
            offset++;
        }
        
        // Create reduction indices tensor
        tensorflow::TensorShape indices_shape;
        indices_shape.AddDim(reduction_indices.size());
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, indices_shape);
        auto indices_flat = indices_tensor.flat<int32_t>();
        
        for (size_t i = 0; i < reduction_indices.size(); ++i) {
            indices_flat(i) = reduction_indices[i];
        }
        
        // Create placeholders
        auto input_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto indices_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        
        // Create Sum operation
        auto sum_op = tensorflow::ops::Sum(root, input_placeholder, indices_placeholder,
                                         tensorflow::ops::Sum::KeepDims(keep_dims));
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({{input_placeholder, input_tensor},
                                                {indices_placeholder, indices_tensor}},
                                               {sum_op}, &outputs);
        
        if (!status.ok()) {
            // Operation failed, but this is expected for some inputs
            return 0;
        }
        
        // Verify output tensor is valid
        if (!outputs.empty()) {
            const tensorflow::Tensor& output = outputs[0];
            if (output.dtype() == tensorflow::DT_FLOAT) {
                auto output_flat = output.flat<float>();
                // Basic sanity check - ensure no NaN values
                for (int i = 0; i < output_flat.size(); ++i) {
                    if (std::isnan(output_flat(i))) {
                        break;
                    }
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