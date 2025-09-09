#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/kernel_def_builder.h>
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/cc/ops/nn_ops.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract dimensions from fuzz data
        int batch = (data[offset] % 4) + 1;
        offset++;
        int height = (data[offset] % 32) + 1;
        offset++;
        int width = (data[offset] % 32) + 1;
        offset++;
        int channels = (data[offset] % 16) + 1;
        offset++;
        
        // Extract kernel size
        int ksize_h = (data[offset] % 8) + 1;
        offset++;
        int ksize_w = (data[offset] % 8) + 1;
        offset++;
        
        // Extract strides
        int stride_h = (data[offset] % 4) + 1;
        offset++;
        int stride_w = (data[offset] % 4) + 1;
        offset++;
        
        // Extract padding type
        bool use_same_padding = data[offset] % 2;
        offset++;
        
        // Extract data type
        tensorflow::DataType input_dtype = tensorflow::DT_FLOAT;
        int dtype_choice = data[offset] % 3;
        offset++;
        switch (dtype_choice) {
            case 0: input_dtype = tensorflow::DT_FLOAT; break;
            case 1: input_dtype = tensorflow::DT_DOUBLE; break;
            case 2: input_dtype = tensorflow::DT_INT32; break;
        }
        
        // Extract argmax type
        tensorflow::DataType argmax_dtype = (data[offset] % 2) ? tensorflow::DT_INT64 : tensorflow::DT_INT32;
        offset++;
        
        // Extract include_batch_in_index
        bool include_batch_in_index = data[offset] % 2;
        offset++;
        
        // Create input tensor
        tensorflow::TensorShape input_shape({batch, height, width, channels});
        tensorflow::Tensor input_tensor(input_dtype, input_shape);
        
        // Fill tensor with fuzz data
        size_t tensor_size = input_tensor.NumElements();
        if (input_dtype == tensorflow::DT_FLOAT) {
            auto flat = input_tensor.flat<float>();
            for (int i = 0; i < tensor_size && offset < size; i++, offset++) {
                flat(i) = static_cast<float>(data[offset]) / 255.0f;
            }
        } else if (input_dtype == tensorflow::DT_DOUBLE) {
            auto flat = input_tensor.flat<double>();
            for (int i = 0; i < tensor_size && offset < size; i++, offset++) {
                flat(i) = static_cast<double>(data[offset]) / 255.0;
            }
        } else if (input_dtype == tensorflow::DT_INT32) {
            auto flat = input_tensor.flat<int32_t>();
            for (int i = 0; i < tensor_size && offset < size; i++, offset++) {
                flat(i) = static_cast<int32_t>(data[offset]);
            }
        }
        
        // Create TensorFlow session
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create placeholder for input
        auto input_placeholder = tensorflow::ops::Placeholder(root, input_dtype);
        
        // Set up MaxPoolWithArgmax operation parameters
        std::vector<int> ksize = {1, ksize_h, ksize_w, 1};
        std::vector<int> strides = {1, stride_h, stride_w, 1};
        std::string padding = use_same_padding ? "SAME" : "VALID";
        
        // Create MaxPoolWithArgmax operation
        auto max_pool_op = tensorflow::ops::MaxPoolWithArgmax(
            root,
            input_placeholder,
            ksize,
            strides,
            padding,
            tensorflow::ops::MaxPoolWithArgmax::Attrs()
                .Targmax(argmax_dtype)
                .IncludeBatchInIndex(include_batch_in_index)
        );
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run(
            {{input_placeholder, input_tensor}},
            {max_pool_op.output, max_pool_op.argmax},
            &outputs
        );
        
        if (!status.ok()) {
            // Operation failed, but this is expected for some invalid inputs
            return 0;
        }
        
        // Verify outputs
        if (outputs.size() == 2) {
            const auto& output_tensor = outputs[0];
            const auto& argmax_tensor = outputs[1];
            
            // Basic sanity checks
            if (output_tensor.dtype() == input_dtype &&
                argmax_tensor.dtype() == argmax_dtype &&
                output_tensor.dims() == 4 &&
                argmax_tensor.dims() == 4 &&
                output_tensor.shape() == argmax_tensor.shape()) {
                // Shapes are consistent
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}