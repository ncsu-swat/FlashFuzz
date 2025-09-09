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
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/const_op.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract dimensions for input tensor
        int batch = (data[offset] % 4) + 1;
        offset++;
        int height = (data[offset] % 32) + 1;
        offset++;
        int width = (data[offset] % 32) + 1;
        offset++;
        int channels = (data[offset] % 8) + 1;
        offset++;
        
        // Extract dimensions for filter tensor
        int filter_height = (data[offset] % 8) + 1;
        offset++;
        int filter_width = (data[offset] % 8) + 1;
        offset++;
        int filter_channels = channels; // Must match input channels
        offset++;
        
        // Extract strides
        int stride_h = (data[offset] % 4) + 1;
        offset++;
        int stride_w = (data[offset] % 4) + 1;
        offset++;
        
        // Extract rates (dilation rates)
        int rate_h = (data[offset] % 4) + 1;
        offset++;
        int rate_w = (data[offset] % 4) + 1;
        offset++;
        
        // Extract padding type
        bool use_same_padding = (data[offset] % 2) == 1;
        offset++;
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input tensor
        tensorflow::TensorShape input_shape({batch, height, width, channels});
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, input_shape);
        auto input_flat = input_tensor.flat<float>();
        
        // Fill input tensor with fuzz data
        for (int i = 0; i < input_flat.size() && offset < size; i++) {
            input_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
            offset++;
        }
        
        // Create filter tensor
        tensorflow::TensorShape filter_shape({filter_height, filter_width, filter_channels});
        tensorflow::Tensor filter_tensor(tensorflow::DT_FLOAT, filter_shape);
        auto filter_flat = filter_tensor.flat<float>();
        
        // Fill filter tensor with fuzz data
        for (int i = 0; i < filter_flat.size() && offset < size; i++) {
            filter_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
            offset++;
        }
        
        // Create constant ops for input and filter
        auto input_op = tensorflow::ops::Const(root, input_tensor);
        auto filter_op = tensorflow::ops::Const(root, filter_tensor);
        
        // Set up attributes for Dilation2D
        std::vector<int> strides = {1, stride_h, stride_w, 1};
        std::vector<int> rates = {1, rate_h, rate_w, 1};
        std::string padding = use_same_padding ? "SAME" : "VALID";
        
        // Create Dilation2D operation
        auto dilation2d = tensorflow::ops::Dilation2D(
            root,
            input_op,
            filter_op,
            tensorflow::ops::Dilation2D::Strides(strides)
                .Rates(rates)
                .Padding(padding)
        );
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({dilation2d}, &outputs);
        
        if (!status.ok()) {
            // Operation failed, but this is acceptable for fuzzing
            return 0;
        }
        
        // Verify output tensor properties
        if (!outputs.empty()) {
            const tensorflow::Tensor& output = outputs[0];
            if (output.dtype() != tensorflow::DT_FLOAT) {
                return -1;
            }
            
            // Check output shape is reasonable
            auto output_shape = output.shape();
            if (output_shape.dims() != 4) {
                return -1;
            }
            
            // Verify output values are finite
            auto output_flat = output.flat<float>();
            for (int i = 0; i < std::min(100, static_cast<int>(output_flat.size())); i++) {
                if (!std::isfinite(output_flat(i))) {
                    return -1;
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