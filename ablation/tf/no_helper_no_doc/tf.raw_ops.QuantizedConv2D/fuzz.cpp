#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/framework/node_def.pb.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/common_runtime/direct_session.h>
#include <tensorflow/core/framework/allocator.h>
#include <tensorflow/core/framework/device_base.h>
#include <tensorflow/core/kernels/ops_util.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract dimensions and parameters from fuzz input
        int32_t batch = (data[offset] % 4) + 1; offset++;
        int32_t input_height = (data[offset] % 32) + 1; offset++;
        int32_t input_width = (data[offset] % 32) + 1; offset++;
        int32_t input_channels = (data[offset] % 16) + 1; offset++;
        
        int32_t filter_height = (data[offset] % 8) + 1; offset++;
        int32_t filter_width = (data[offset] % 8) + 1; offset++;
        int32_t output_channels = (data[offset] % 16) + 1; offset++;
        
        int32_t stride_h = (data[offset] % 4) + 1; offset++;
        int32_t stride_w = (data[offset] % 4) + 1; offset++;
        
        bool use_same_padding = (data[offset] % 2) == 1; offset++;
        
        if (offset + 6 > size) return 0;
        
        // Create input tensor
        tensorflow::TensorShape input_shape({batch, input_height, input_width, input_channels});
        tensorflow::Tensor input_tensor(tensorflow::DT_QUINT8, input_shape);
        auto input_flat = input_tensor.flat<tensorflow::quint8>();
        
        // Fill input with fuzz data
        for (int i = 0; i < input_flat.size() && offset < size; ++i) {
            input_flat(i) = tensorflow::quint8(data[offset % size]);
            offset++;
        }
        
        // Create filter tensor
        tensorflow::TensorShape filter_shape({filter_height, filter_width, input_channels, output_channels});
        tensorflow::Tensor filter_tensor(tensorflow::DT_QUINT8, filter_shape);
        auto filter_flat = filter_tensor.flat<tensorflow::quint8>();
        
        // Fill filter with fuzz data
        for (int i = 0; i < filter_flat.size() && offset < size; ++i) {
            filter_flat(i) = tensorflow::quint8(data[offset % size]);
            offset++;
        }
        
        // Create min/max tensors
        tensorflow::Tensor min_input(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor max_input(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor min_filter(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor max_filter(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        
        min_input.scalar<float>()() = -128.0f;
        max_input.scalar<float>()() = 127.0f;
        min_filter.scalar<float>()() = -128.0f;
        max_filter.scalar<float>()() = 127.0f;
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node = graph_def.add_node();
        node->set_name("quantized_conv2d");
        node->set_op("QuantizedConv2D");
        
        // Add input names
        node->add_input("input");
        node->add_input("filter");
        node->add_input("min_input");
        node->add_input("max_input");
        node->add_input("min_filter");
        node->add_input("max_filter");
        
        // Set attributes
        tensorflow::AttrValue strides_attr;
        auto* strides_list = strides_attr.mutable_list();
        strides_list->add_i(1);
        strides_list->add_i(stride_h);
        strides_list->add_i(stride_w);
        strides_list->add_i(1);
        (*node->mutable_attr())["strides"] = strides_attr;
        
        tensorflow::AttrValue padding_attr;
        padding_attr.set_s(use_same_padding ? "SAME" : "VALID");
        (*node->mutable_attr())["padding"] = padding_attr;
        
        tensorflow::AttrValue tinput_attr;
        tinput_attr.set_type(tensorflow::DT_QUINT8);
        (*node->mutable_attr())["Tinput"] = tinput_attr;
        
        tensorflow::AttrValue tfilter_attr;
        tfilter_attr.set_type(tensorflow::DT_QUINT8);
        (*node->mutable_attr())["Tfilter"] = tfilter_attr;
        
        tensorflow::AttrValue out_type_attr;
        out_type_attr.set_type(tensorflow::DT_QINT32);
        (*node->mutable_attr())["out_type"] = out_type_attr;
        
        // Add placeholder nodes
        auto add_placeholder = [&](const std::string& name, tensorflow::DataType dtype, const tensorflow::TensorShape& shape) {
            tensorflow::NodeDef* placeholder = graph_def.add_node();
            placeholder->set_name(name);
            placeholder->set_op("Placeholder");
            tensorflow::AttrValue dtype_attr;
            dtype_attr.set_type(dtype);
            (*placeholder->mutable_attr())["dtype"] = dtype_attr;
            
            tensorflow::AttrValue shape_attr;
            auto* shape_proto = shape_attr.mutable_shape();
            for (int i = 0; i < shape.dims(); ++i) {
                shape_proto->add_dim()->set_size(shape.dim_size(i));
            }
            (*placeholder->mutable_attr())["shape"] = shape_attr;
        };
        
        add_placeholder("input", tensorflow::DT_QUINT8, input_shape);
        add_placeholder("filter", tensorflow::DT_QUINT8, filter_shape);
        add_placeholder("min_input", tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        add_placeholder("max_input", tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        add_placeholder("min_filter", tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        add_placeholder("max_filter", tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        
        // Create session and run
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"input", input_tensor},
            {"filter", filter_tensor},
            {"min_input", min_input},
            {"max_input", max_input},
            {"min_filter", min_filter},
            {"max_filter", max_filter}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {"quantized_conv2d:0", "quantized_conv2d:1", "quantized_conv2d:2"};
        
        status = session->Run(inputs, output_names, {}, &outputs);
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}