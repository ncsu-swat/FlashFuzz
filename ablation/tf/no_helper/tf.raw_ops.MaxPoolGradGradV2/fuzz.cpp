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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract dimensions from fuzz data
        int batch = (data[offset] % 4) + 1;
        offset++;
        int height = (data[offset] % 8) + 2;
        offset++;
        int width = (data[offset] % 8) + 2;
        offset++;
        int channels = (data[offset] % 4) + 1;
        offset++;
        
        // Extract pooling parameters
        int ksize_h = (data[offset] % 3) + 1;
        offset++;
        int ksize_w = (data[offset] % 3) + 1;
        offset++;
        int stride_h = (data[offset] % 2) + 1;
        offset++;
        int stride_w = (data[offset] % 2) + 1;
        offset++;
        
        // Extract padding and data format
        bool use_same_padding = (data[offset] % 2) == 0;
        offset++;
        bool use_nchw = (data[offset] % 2) == 0;
        offset++;
        
        // Calculate output dimensions
        int out_height, out_width;
        if (use_same_padding) {
            out_height = (height + stride_h - 1) / stride_h;
            out_width = (width + stride_w - 1) / stride_w;
        } else {
            out_height = (height - ksize_h) / stride_h + 1;
            out_width = (width - ksize_w) / stride_w + 1;
        }
        
        if (out_height <= 0 || out_width <= 0) return 0;
        
        // Create input tensors
        tensorflow::TensorShape input_shape, output_shape;
        if (use_nchw) {
            input_shape = tensorflow::TensorShape({batch, channels, height, width});
            output_shape = tensorflow::TensorShape({batch, channels, out_height, out_width});
        } else {
            input_shape = tensorflow::TensorShape({batch, height, width, channels});
            output_shape = tensorflow::TensorShape({batch, out_height, out_width, channels});
        }
        
        tensorflow::Tensor orig_input(tensorflow::DT_FLOAT, input_shape);
        tensorflow::Tensor orig_output(tensorflow::DT_FLOAT, output_shape);
        tensorflow::Tensor grad(tensorflow::DT_FLOAT, input_shape);
        
        // Fill tensors with fuzz data
        auto orig_input_flat = orig_input.flat<float>();
        auto orig_output_flat = orig_output.flat<float>();
        auto grad_flat = grad.flat<float>();
        
        size_t total_input_elements = orig_input_flat.size();
        size_t total_output_elements = orig_output_flat.size();
        size_t total_grad_elements = grad_flat.size();
        
        for (int i = 0; i < total_input_elements && offset < size; i++) {
            orig_input_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
            offset++;
        }
        
        for (int i = 0; i < total_output_elements && offset < size; i++) {
            orig_output_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
            offset++;
        }
        
        for (int i = 0; i < total_grad_elements && offset < size; i++) {
            grad_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
            offset++;
        }
        
        // Create ksize and strides tensors
        tensorflow::Tensor ksize(tensorflow::DT_INT32, tensorflow::TensorShape({4}));
        tensorflow::Tensor strides(tensorflow::DT_INT32, tensorflow::TensorShape({4}));
        
        auto ksize_flat = ksize.flat<int32_t>();
        auto strides_flat = strides.flat<int32_t>();
        
        if (use_nchw) {
            ksize_flat(0) = 1; ksize_flat(1) = 1; ksize_flat(2) = ksize_h; ksize_flat(3) = ksize_w;
            strides_flat(0) = 1; strides_flat(1) = 1; strides_flat(2) = stride_h; strides_flat(3) = stride_w;
        } else {
            ksize_flat(0) = 1; ksize_flat(1) = ksize_h; ksize_flat(2) = ksize_w; ksize_flat(3) = 1;
            strides_flat(0) = 1; strides_flat(1) = stride_h; strides_flat(2) = stride_w; strides_flat(3) = 1;
        }
        
        // Create session and graph
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        tensorflow::GraphDef graph_def;
        tensorflow::GraphDefBuilder builder(tensorflow::GraphDefBuilder::kFailImmediately);
        
        // Add placeholders
        auto orig_input_node = tensorflow::ops::Placeholder(builder.opts()
            .WithName("orig_input")
            .WithAttr("dtype", tensorflow::DT_FLOAT)
            .WithAttr("shape", input_shape));
            
        auto orig_output_node = tensorflow::ops::Placeholder(builder.opts()
            .WithName("orig_output")
            .WithAttr("dtype", tensorflow::DT_FLOAT)
            .WithAttr("shape", output_shape));
            
        auto grad_node = tensorflow::ops::Placeholder(builder.opts()
            .WithName("grad")
            .WithAttr("dtype", tensorflow::DT_FLOAT)
            .WithAttr("shape", input_shape));
            
        auto ksize_node = tensorflow::ops::Placeholder(builder.opts()
            .WithName("ksize")
            .WithAttr("dtype", tensorflow::DT_INT32)
            .WithAttr("shape", tensorflow::TensorShape({4})));
            
        auto strides_node = tensorflow::ops::Placeholder(builder.opts()
            .WithName("strides")
            .WithAttr("dtype", tensorflow::DT_INT32)
            .WithAttr("shape", tensorflow::TensorShape({4})));
        
        // Add MaxPoolGradGradV2 operation
        auto max_pool_grad_grad = tensorflow::ops::UnaryOp("MaxPoolGradGradV2", 
            {orig_input_node, orig_output_node, grad_node, ksize_node, strides_node},
            builder.opts()
                .WithName("max_pool_grad_grad")
                .WithAttr("padding", use_same_padding ? "SAME" : "VALID")
                .WithAttr("data_format", use_nchw ? "NCHW" : "NHWC"));
        
        tensorflow::Status status = builder.ToGraphDef(&graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run the operation
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"orig_input", orig_input},
            {"orig_output", orig_output},
            {"grad", grad},
            {"ksize", ksize},
            {"strides", strides}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"max_pool_grad_grad"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Successfully executed the operation
            auto output_flat = outputs[0].flat<float>();
            volatile float sum = 0.0f;
            for (int i = 0; i < output_flat.size(); i++) {
                sum += output_flat(i);
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