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
        
        // Extract dimensions for input tensor (5D: batch, depth, height, width, channels)
        int batch = (data[offset] % 4) + 1;
        offset++;
        int depth = (data[offset] % 8) + 1;
        offset++;
        int height = (data[offset] % 16) + 1;
        offset++;
        int width = (data[offset] % 16) + 1;
        offset++;
        int channels = (data[offset] % 8) + 1;
        offset++;
        
        // Extract kernel size (3 values for 3D)
        int ksize_d = (data[offset] % 4) + 1;
        offset++;
        int ksize_h = (data[offset] % 4) + 1;
        offset++;
        int ksize_w = (data[offset] % 4) + 1;
        offset++;
        
        // Extract strides (3 values for 3D)
        int stride_d = (data[offset] % 3) + 1;
        offset++;
        int stride_h = (data[offset] % 3) + 1;
        offset++;
        int stride_w = (data[offset] % 3) + 1;
        offset++;
        
        // Extract padding type
        bool use_valid_padding = (data[offset] % 2) == 0;
        offset++;
        
        // Extract data format
        bool use_ndhwc = (data[offset] % 2) == 0;
        offset++;
        
        // Create input tensor
        tensorflow::TensorShape input_shape;
        if (use_ndhwc) {
            input_shape = tensorflow::TensorShape({batch, depth, height, width, channels});
        } else {
            input_shape = tensorflow::TensorShape({batch, channels, depth, height, width});
        }
        
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, input_shape);
        auto input_flat = input_tensor.flat<float>();
        
        // Fill input tensor with fuzz data
        size_t tensor_size = input_flat.size();
        for (int i = 0; i < tensor_size && offset < size; ++i) {
            input_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
            offset++;
        }
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Build graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* input_node = graph_def.add_node();
        input_node->set_name("input");
        input_node->set_op("Placeholder");
        (*input_node->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        input_node->mutable_attr()->insert({"shape", tensorflow::AttrValue()});
        
        tensorflow::NodeDef* maxpool_node = graph_def.add_node();
        maxpool_node->set_name("maxpool3d");
        maxpool_node->set_op("MaxPool3D");
        maxpool_node->add_input("input");
        
        // Set ksize attribute
        auto ksize_attr = maxpool_node->mutable_attr()->insert({"ksize", tensorflow::AttrValue()}).first;
        ksize_attr->second.mutable_list()->add_i(1);  // batch
        if (use_ndhwc) {
            ksize_attr->second.mutable_list()->add_i(ksize_d);
            ksize_attr->second.mutable_list()->add_i(ksize_h);
            ksize_attr->second.mutable_list()->add_i(ksize_w);
            ksize_attr->second.mutable_list()->add_i(1);  // channels
        } else {
            ksize_attr->second.mutable_list()->add_i(1);  // channels
            ksize_attr->second.mutable_list()->add_i(ksize_d);
            ksize_attr->second.mutable_list()->add_i(ksize_h);
            ksize_attr->second.mutable_list()->add_i(ksize_w);
        }
        
        // Set strides attribute
        auto strides_attr = maxpool_node->mutable_attr()->insert({"strides", tensorflow::AttrValue()}).first;
        strides_attr->second.mutable_list()->add_i(1);  // batch
        if (use_ndhwc) {
            strides_attr->second.mutable_list()->add_i(stride_d);
            strides_attr->second.mutable_list()->add_i(stride_h);
            strides_attr->second.mutable_list()->add_i(stride_w);
            strides_attr->second.mutable_list()->add_i(1);  // channels
        } else {
            strides_attr->second.mutable_list()->add_i(1);  // channels
            strides_attr->second.mutable_list()->add_i(stride_d);
            strides_attr->second.mutable_list()->add_i(stride_h);
            strides_attr->second.mutable_list()->add_i(stride_w);
        }
        
        // Set padding attribute
        auto padding_attr = maxpool_node->mutable_attr()->insert({"padding", tensorflow::AttrValue()}).first;
        padding_attr->second.set_s(use_valid_padding ? "VALID" : "SAME");
        
        // Set data_format attribute
        auto data_format_attr = maxpool_node->mutable_attr()->insert({"data_format", tensorflow::AttrValue()}).first;
        data_format_attr->second.set_s(use_ndhwc ? "NDHWC" : "NCDHW");
        
        // Set T attribute
        auto t_attr = maxpool_node->mutable_attr()->insert({"T", tensorflow::AttrValue()}).first;
        t_attr->second.set_type(tensorflow::DT_FLOAT);
        
        // Create session and run
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({{"input", input_tensor}}, {"maxpool3d"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Successfully executed MaxPool3D
            auto output_flat = outputs[0].flat<float>();
            volatile float sum = 0.0f;
            for (int i = 0; i < output_flat.size(); ++i) {
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