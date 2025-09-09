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
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/framework/fake_input.h>
#include <tensorflow/core/kernels/ops_testutil.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract dimensions for input tensor
        int batch_size = (data[offset] % 4) + 1;
        offset++;
        int height = (data[offset] % 32) + 1;
        offset++;
        int width = (data[offset] % 32) + 1;
        offset++;
        int channels = (data[offset] % 16) + 1;
        offset++;
        
        // Extract pooling parameters
        int ksize_h = (data[offset] % 8) + 1;
        offset++;
        int ksize_w = (data[offset] % 8) + 1;
        offset++;
        int stride_h = (data[offset] % 4) + 1;
        offset++;
        int stride_w = (data[offset] % 4) + 1;
        offset++;
        
        // Extract padding type
        bool use_same_padding = (data[offset] % 2) == 1;
        offset++;
        
        // Extract data type
        tensorflow::DataType dtype = (data[offset] % 2) == 0 ? 
            tensorflow::DT_FLOAT : tensorflow::DT_HALF;
        offset++;
        
        // Extract Targmax type
        tensorflow::DataType targmax_type = (data[offset] % 2) == 0 ? 
            tensorflow::DT_INT32 : tensorflow::DT_INT64;
        offset++;
        
        // Create input tensor shape
        tensorflow::TensorShape input_shape({batch_size, height, width, channels});
        
        // Create input tensor
        tensorflow::Tensor input_tensor(dtype, input_shape);
        
        // Fill tensor with fuzz data
        if (dtype == tensorflow::DT_FLOAT) {
            auto flat = input_tensor.flat<float>();
            for (int i = 0; i < flat.size() && offset < size - 4; i++) {
                float val;
                memcpy(&val, data + offset, sizeof(float));
                flat(i) = val;
                offset += sizeof(float);
            }
        } else {
            auto flat = input_tensor.flat<Eigen::half>();
            for (int i = 0; i < flat.size() && offset < size - 2; i++) {
                uint16_t val;
                memcpy(&val, data + offset, sizeof(uint16_t));
                flat(i) = Eigen::half_impl::raw_uint16_to_half(val);
                offset += sizeof(uint16_t);
            }
        }
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Build graph
        tensorflow::GraphDefBuilder builder(tensorflow::GraphDefBuilder::kFailImmediately);
        
        tensorflow::Node* input_node = tensorflow::ops::Placeholder(
            builder.opts().WithName("input").WithAttr("dtype", dtype));
        
        std::vector<int> ksize = {1, ksize_h, ksize_w, 1};
        std::vector<int> strides = {1, stride_h, stride_w, 1};
        std::string padding = use_same_padding ? "SAME" : "VALID";
        
        tensorflow::Node* maxpool_node = tensorflow::ops::MaxPoolWithArgmax(
            input_node,
            builder.opts()
                .WithName("maxpool")
                .WithAttr("ksize", ksize)
                .WithAttr("strides", strides)
                .WithAttr("padding", padding)
                .WithAttr("Targmax", targmax_type));
        
        tensorflow::GraphDef graph_def;
        tensorflow::Status status = builder.ToGraphDef(&graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({{"input", input_tensor}}, 
                             {"maxpool:0", "maxpool:1"}, 
                             {}, &outputs);
        
        if (status.ok() && outputs.size() == 2) {
            // Verify output shapes are reasonable
            const auto& output_shape = outputs[0].shape();
            const auto& argmax_shape = outputs[1].shape();
            
            if (output_shape.dims() == 4 && argmax_shape.dims() == 4 &&
                output_shape.dim_size(0) == batch_size &&
                output_shape.dim_size(3) == channels &&
                argmax_shape.dim_size(0) == batch_size &&
                argmax_shape.dim_size(3) == channels) {
                // Valid output
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