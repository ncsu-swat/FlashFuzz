#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/node_def_builder.h>
#include <tensorflow/core/framework/fake_input.h>
#include <tensorflow/core/kernels/ops_testutil.h>
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>

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
        int channels = (data[offset] % 8) + 1;
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
        
        // Extract padding type (0 = VALID, 1 = SAME)
        bool use_same_padding = (data[offset] % 2) == 1;
        offset++;
        
        // Extract data format (0 = NHWC, 1 = NCHW)
        bool use_nchw = (data[offset] % 2) == 1;
        offset++;
        
        // Create input tensor
        tensorflow::TensorShape input_shape;
        if (use_nchw) {
            input_shape = tensorflow::TensorShape({batch_size, channels, height, width});
        } else {
            input_shape = tensorflow::TensorShape({batch_size, height, width, channels});
        }
        
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, input_shape);
        auto input_flat = input_tensor.flat<float>();
        
        // Fill input tensor with fuzz data
        size_t tensor_size = input_flat.size();
        for (int i = 0; i < tensor_size && offset < size; ++i) {
            // Convert uint8 to float in range [-10, 10]
            input_flat(i) = (static_cast<float>(data[offset % size]) - 128.0f) / 12.8f;
            offset++;
        }
        
        // Create ksize tensor
        tensorflow::Tensor ksize_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({4}));
        auto ksize_flat = ksize_tensor.flat<int32_t>();
        if (use_nchw) {
            ksize_flat(0) = 1;  // batch
            ksize_flat(1) = 1;  // channels
            ksize_flat(2) = ksize_h;  // height
            ksize_flat(3) = ksize_w;  // width
        } else {
            ksize_flat(0) = 1;  // batch
            ksize_flat(1) = ksize_h;  // height
            ksize_flat(2) = ksize_w;  // width
            ksize_flat(3) = 1;  // channels
        }
        
        // Create strides tensor
        tensorflow::Tensor strides_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({4}));
        auto strides_flat = strides_tensor.flat<int32_t>();
        if (use_nchw) {
            strides_flat(0) = 1;  // batch
            strides_flat(1) = 1;  // channels
            strides_flat(2) = stride_h;  // height
            strides_flat(3) = stride_w;  // width
        } else {
            strides_flat(0) = 1;  // batch
            strides_flat(1) = stride_h;  // height
            strides_flat(2) = stride_w;  // width
            strides_flat(3) = 1;  // channels
        }
        
        // Create session and graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* input_node = graph_def.add_node();
        input_node->set_name("input");
        input_node->set_op("Placeholder");
        (*input_node->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        
        tensorflow::NodeDef* ksize_node = graph_def.add_node();
        ksize_node->set_name("ksize");
        ksize_node->set_op("Const");
        (*ksize_node->mutable_attr())["dtype"].set_type(tensorflow::DT_INT32);
        
        tensorflow::NodeDef* strides_node = graph_def.add_node();
        strides_node->set_name("strides");
        strides_node->set_op("Const");
        (*strides_node->mutable_attr())["dtype"].set_type(tensorflow::DT_INT32);
        
        tensorflow::NodeDef* maxpool_node = graph_def.add_node();
        maxpool_node->set_name("maxpool");
        maxpool_node->set_op("MaxPoolV2");
        maxpool_node->add_input("input");
        maxpool_node->add_input("ksize");
        maxpool_node->add_input("strides");
        (*maxpool_node->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
        (*maxpool_node->mutable_attr())["padding"].set_s(use_same_padding ? "SAME" : "VALID");
        (*maxpool_node->mutable_attr())["data_format"].set_s(use_nchw ? "NCHW" : "NHWC");
        
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        if (!session) return 0;
        
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) return 0;
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({{"input", input_tensor}, {"ksize", ksize_tensor}, {"strides", strides_tensor}}, 
                             {"maxpool"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Verify output tensor is valid
            const tensorflow::Tensor& output = outputs[0];
            if (output.dtype() == tensorflow::DT_FLOAT && output.dims() == 4) {
                auto output_flat = output.flat<float>();
                // Basic sanity check - ensure no NaN values
                for (int i = 0; i < std::min(100, static_cast<int>(output_flat.size())); ++i) {
                    if (std::isnan(output_flat(i))) {
                        break;
                    }
                }
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