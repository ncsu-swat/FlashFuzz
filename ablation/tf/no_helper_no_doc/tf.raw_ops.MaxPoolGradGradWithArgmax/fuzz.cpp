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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract dimensions and parameters from fuzz input
        int batch_size = (data[offset] % 4) + 1;
        offset++;
        int height = (data[offset] % 8) + 2;
        offset++;
        int width = (data[offset] % 8) + 2;
        offset++;
        int channels = (data[offset] % 4) + 1;
        offset++;
        
        int ksize_h = (data[offset] % 3) + 1;
        offset++;
        int ksize_w = (data[offset] % 3) + 1;
        offset++;
        
        int stride_h = (data[offset] % 2) + 1;
        offset++;
        int stride_w = (data[offset] % 2) + 1;
        offset++;
        
        // Calculate output dimensions for max pool
        int out_height = (height - ksize_h) / stride_h + 1;
        int out_width = (width - ksize_w) / stride_w + 1;
        
        if (out_height <= 0 || out_width <= 0) return 0;
        
        // Create input tensor
        tensorflow::TensorShape input_shape({batch_size, height, width, channels});
        tensorflow::Tensor input(tensorflow::DT_FLOAT, input_shape);
        auto input_flat = input.flat<float>();
        
        // Fill input with fuzz data
        for (int i = 0; i < input_flat.size() && offset < size - 4; i++) {
            float val;
            memcpy(&val, data + offset, sizeof(float));
            input_flat(i) = val;
            offset += sizeof(float);
            if (offset >= size - 4) break;
        }
        
        // Create grad tensor (gradient from next layer)
        tensorflow::TensorShape grad_shape({batch_size, out_height, out_width, channels});
        tensorflow::Tensor grad(tensorflow::DT_FLOAT, grad_shape);
        auto grad_flat = grad.flat<float>();
        
        // Fill grad with remaining fuzz data
        for (int i = 0; i < grad_flat.size() && offset < size - 4; i++) {
            float val;
            memcpy(&val, data + offset, sizeof(float));
            grad_flat(i) = val;
            offset += sizeof(float);
            if (offset >= size - 4) break;
        }
        
        // Create argmax tensor (indices from max pooling)
        tensorflow::Tensor argmax(tensorflow::DT_INT64, grad_shape);
        auto argmax_flat = argmax.flat<int64_t>();
        
        // Fill argmax with valid indices
        for (int i = 0; i < argmax_flat.size(); i++) {
            int max_idx = ksize_h * ksize_w - 1;
            argmax_flat(i) = (offset < size) ? (data[offset] % (max_idx + 1)) : 0;
            if (offset < size) offset++;
        }
        
        // Create session and graph
        tensorflow::GraphDefBuilder builder(tensorflow::GraphDefBuilder::kFailImmediately);
        
        // Add placeholders
        auto input_node = tensorflow::ops::Placeholder(builder.opts()
            .WithName("input")
            .WithAttr("dtype", tensorflow::DT_FLOAT)
            .WithAttr("shape", input_shape));
            
        auto grad_node = tensorflow::ops::Placeholder(builder.opts()
            .WithName("grad")
            .WithAttr("dtype", tensorflow::DT_FLOAT)
            .WithAttr("shape", grad_shape));
            
        auto argmax_node = tensorflow::ops::Placeholder(builder.opts()
            .WithName("argmax")
            .WithAttr("dtype", tensorflow::DT_INT64)
            .WithAttr("shape", grad_shape));
        
        // Create ksize, strides, padding attributes
        std::vector<int> ksize = {1, ksize_h, ksize_w, 1};
        std::vector<int> strides = {1, stride_h, stride_w, 1};
        std::string padding = "VALID";
        
        // Add MaxPoolGradGradWithArgmax operation
        auto maxpool_grad_grad = tensorflow::ops::MaxPoolGradGradWithArgmax(
            input_node, grad_node, argmax_node,
            builder.opts()
                .WithName("maxpool_grad_grad")
                .WithAttr("ksize", ksize)
                .WithAttr("strides", strides)
                .WithAttr("padding", padding));
        
        tensorflow::GraphDef graph_def;
        if (!builder.ToGraphDef(&graph_def).ok()) {
            return 0;
        }
        
        // Create session
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        if (!session->Create(graph_def).ok()) {
            return 0;
        }
        
        // Prepare inputs
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"input", input},
            {"grad", grad},
            {"argmax", argmax}
        };
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session->Run(inputs, {"maxpool_grad_grad"}, {}, &outputs);
        
        // Clean up
        session->Close();
        
        if (!status.ok()) {
            return 0;
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}