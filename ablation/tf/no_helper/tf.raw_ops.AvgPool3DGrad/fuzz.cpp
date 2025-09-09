#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/nn_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 64) return 0;
        
        // Extract basic parameters
        int batch = (data[offset] % 4) + 1;
        offset++;
        int depth = (data[offset] % 8) + 2;
        offset++;
        int height = (data[offset] % 8) + 2;
        offset++;
        int width = (data[offset] % 8) + 2;
        offset++;
        int channels = (data[offset] % 4) + 1;
        offset++;
        
        // Extract kernel size (must have ksize[0] = ksize[4] = 1)
        int ksize_d = (data[offset] % 3) + 1;
        offset++;
        int ksize_h = (data[offset] % 3) + 1;
        offset++;
        int ksize_w = (data[offset] % 3) + 1;
        offset++;
        
        // Extract strides (must have strides[0] = strides[4] = 1)
        int stride_d = (data[offset] % 2) + 1;
        offset++;
        int stride_h = (data[offset] % 2) + 1;
        offset++;
        int stride_w = (data[offset] % 2) + 1;
        offset++;
        
        // Extract padding type
        bool use_same_padding = (data[offset] % 2) == 0;
        offset++;
        
        // Extract data format
        bool use_ndhwc = (data[offset] % 2) == 0;
        offset++;
        
        // Calculate output dimensions for grad tensor
        int out_depth, out_height, out_width;
        if (use_same_padding) {
            out_depth = (depth + stride_d - 1) / stride_d;
            out_height = (height + stride_h - 1) / stride_h;
            out_width = (width + stride_w - 1) / stride_w;
        } else {
            out_depth = (depth - ksize_d) / stride_d + 1;
            out_height = (height - ksize_h) / stride_h + 1;
            out_width = (width - ksize_w) / stride_w + 1;
        }
        
        if (out_depth <= 0 || out_height <= 0 || out_width <= 0) return 0;
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create orig_input_shape tensor
        tensorflow::Tensor orig_input_shape_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({5}));
        auto orig_input_flat = orig_input_shape_tensor.flat<int32_t>();
        if (use_ndhwc) {
            orig_input_flat(0) = batch;
            orig_input_flat(1) = depth;
            orig_input_flat(2) = height;
            orig_input_flat(3) = width;
            orig_input_flat(4) = channels;
        } else {
            orig_input_flat(0) = batch;
            orig_input_flat(1) = channels;
            orig_input_flat(2) = depth;
            orig_input_flat(3) = height;
            orig_input_flat(4) = width;
        }
        
        // Create grad tensor
        tensorflow::TensorShape grad_shape;
        if (use_ndhwc) {
            grad_shape = tensorflow::TensorShape({batch, out_depth, out_height, out_width, channels});
        } else {
            grad_shape = tensorflow::TensorShape({batch, channels, out_depth, out_height, out_width});
        }
        
        tensorflow::Tensor grad_tensor(tensorflow::DT_FLOAT, grad_shape);
        auto grad_flat = grad_tensor.flat<float>();
        
        // Fill grad tensor with fuzz data
        size_t grad_size = grad_flat.size();
        for (size_t i = 0; i < grad_size && offset < size; i++) {
            grad_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
            offset++;
        }
        
        // Create input placeholders
        auto orig_input_shape_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto grad_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        // Set up operation attributes
        std::vector<int> ksize = {1, ksize_d, ksize_h, ksize_w, 1};
        std::vector<int> strides = {1, stride_d, stride_h, stride_w, 1};
        std::string padding = use_same_padding ? "SAME" : "VALID";
        std::string data_format = use_ndhwc ? "NDHWC" : "NCDHW";
        
        // Create AvgPool3DGrad operation
        auto avg_pool_3d_grad = tensorflow::ops::AvgPool3DGrad(
            root,
            orig_input_shape_ph,
            grad_ph,
            ksize,
            strides,
            padding,
            tensorflow::ops::AvgPool3DGrad::DataFormat(data_format)
        );
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run(
            {{orig_input_shape_ph, orig_input_shape_tensor}, {grad_ph, grad_tensor}},
            {avg_pool_3d_grad},
            &outputs
        );
        
        if (!status.ok()) {
            std::cout << "Operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output tensor properties
        if (!outputs.empty()) {
            const auto& output = outputs[0];
            if (output.dtype() != tensorflow::DT_FLOAT) {
                std::cout << "Unexpected output dtype" << std::endl;
                return 0;
            }
            
            // Check output shape matches original input shape
            auto expected_shape = tensorflow::TensorShape();
            for (int i = 0; i < 5; i++) {
                expected_shape.AddDim(orig_input_flat(i));
            }
            
            if (output.shape() != expected_shape) {
                std::cout << "Output shape mismatch" << std::endl;
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