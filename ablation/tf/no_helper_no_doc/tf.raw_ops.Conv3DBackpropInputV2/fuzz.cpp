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
        
        if (size < 64) return 0;
        
        // Extract dimensions from fuzz data
        int32_t batch = (data[offset] % 4) + 1; offset++;
        int32_t in_depth = (data[offset] % 8) + 1; offset++;
        int32_t in_height = (data[offset] % 16) + 1; offset++;
        int32_t in_width = (data[offset] % 16) + 1; offset++;
        int32_t in_channels = (data[offset] % 8) + 1; offset++;
        
        int32_t filter_depth = (data[offset] % 4) + 1; offset++;
        int32_t filter_height = (data[offset] % 4) + 1; offset++;
        int32_t filter_width = (data[offset] % 4) + 1; offset++;
        int32_t out_channels = (data[offset] % 8) + 1; offset++;
        
        int32_t out_batch = batch;
        int32_t out_depth = (data[offset] % 8) + 1; offset++;
        int32_t out_height = (data[offset] % 8) + 1; offset++;
        int32_t out_width = (data[offset] % 8) + 1; offset++;
        
        // Create input_sizes tensor (5D: [batch, depth, height, width, channels])
        tensorflow::Tensor input_sizes(tensorflow::DT_INT32, tensorflow::TensorShape({5}));
        auto input_sizes_flat = input_sizes.flat<int32_t>();
        input_sizes_flat(0) = batch;
        input_sizes_flat(1) = in_depth;
        input_sizes_flat(2) = in_height;
        input_sizes_flat(3) = in_width;
        input_sizes_flat(4) = in_channels;
        
        // Create filter tensor
        tensorflow::TensorShape filter_shape({filter_depth, filter_height, filter_width, in_channels, out_channels});
        tensorflow::Tensor filter(tensorflow::DT_FLOAT, filter_shape);
        auto filter_flat = filter.flat<float>();
        
        // Fill filter with fuzz data
        size_t filter_size = filter_flat.size();
        for (size_t i = 0; i < filter_size && offset < size; ++i, ++offset) {
            filter_flat(i) = static_cast<float>(data[offset % size]) / 255.0f - 0.5f;
        }
        
        // Create out_backprop tensor
        tensorflow::TensorShape out_backprop_shape({out_batch, out_depth, out_height, out_width, out_channels});
        tensorflow::Tensor out_backprop(tensorflow::DT_FLOAT, out_backprop_shape);
        auto out_backprop_flat = out_backprop.flat<float>();
        
        // Fill out_backprop with fuzz data
        size_t out_backprop_size = out_backprop_flat.size();
        for (size_t i = 0; i < out_backprop_size && offset < size; ++i, ++offset) {
            out_backprop_flat(i) = static_cast<float>(data[offset % size]) / 255.0f - 0.5f;
        }
        
        // Create session and graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node_def = graph_def.add_node();
        
        node_def->set_name("conv3d_backprop_input");
        node_def->set_op("Conv3DBackpropInputV2");
        node_def->add_input("input_sizes");
        node_def->add_input("filter");
        node_def->add_input("out_backprop");
        
        // Set attributes
        tensorflow::AttrValue attr_T;
        attr_T.set_type(tensorflow::DT_FLOAT);
        (*node_def->mutable_attr())["T"] = attr_T;
        
        tensorflow::AttrValue attr_strides;
        attr_strides.mutable_list()->add_i(1);
        attr_strides.mutable_list()->add_i(1);
        attr_strides.mutable_list()->add_i(1);
        attr_strides.mutable_list()->add_i(1);
        attr_strides.mutable_list()->add_i(1);
        (*node_def->mutable_attr())["strides"] = attr_strides;
        
        tensorflow::AttrValue attr_padding;
        attr_padding.set_s(data[offset % size] % 2 == 0 ? "VALID" : "SAME");
        (*node_def->mutable_attr())["padding"] = attr_padding;
        
        // Add input nodes
        tensorflow::NodeDef* input_sizes_node = graph_def.add_node();
        input_sizes_node->set_name("input_sizes");
        input_sizes_node->set_op("Const");
        tensorflow::AttrValue input_sizes_attr;
        input_sizes_attr.set_type(tensorflow::DT_INT32);
        (*input_sizes_node->mutable_attr())["dtype"] = input_sizes_attr;
        
        tensorflow::NodeDef* filter_node = graph_def.add_node();
        filter_node->set_name("filter");
        filter_node->set_op("Const");
        tensorflow::AttrValue filter_attr;
        filter_attr.set_type(tensorflow::DT_FLOAT);
        (*filter_node->mutable_attr())["dtype"] = filter_attr;
        
        tensorflow::NodeDef* out_backprop_node = graph_def.add_node();
        out_backprop_node->set_name("out_backprop");
        out_backprop_node->set_op("Const");
        tensorflow::AttrValue out_backprop_attr;
        out_backprop_attr.set_type(tensorflow::DT_FLOAT);
        (*out_backprop_node->mutable_attr())["dtype"] = out_backprop_attr;
        
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        if (!session) return 0;
        
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) return 0;
        
        // Run the operation
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"input_sizes", input_sizes},
            {"filter", filter},
            {"out_backprop", out_backprop}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"conv3d_backprop_input"}, {}, &outputs);
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}