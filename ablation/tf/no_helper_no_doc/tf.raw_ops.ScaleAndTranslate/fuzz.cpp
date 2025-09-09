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
        
        // Extract dimensions and parameters from fuzz input
        int batch_size = (data[offset] % 4) + 1;
        offset++;
        int height = (data[offset] % 64) + 8;
        offset++;
        int width = (data[offset] % 64) + 8;
        offset++;
        int channels = (data[offset] % 4) + 1;
        offset++;
        
        // Extract scale values
        float scale_x = *reinterpret_cast<const float*>(data + offset) * 0.1f + 1.0f;
        offset += 4;
        float scale_y = *reinterpret_cast<const float*>(data + offset) * 0.1f + 1.0f;
        offset += 4;
        
        // Extract translation values
        float translate_x = *reinterpret_cast<const float*>(data + offset) * 0.1f;
        offset += 4;
        float translate_y = *reinterpret_cast<const float*>(data + offset) * 0.1f;
        offset += 4;
        
        // Extract interpolation method (0 or 1)
        bool antialias = (data[offset] % 2) == 1;
        offset++;
        
        // Create input tensor
        tensorflow::Tensor images(tensorflow::DT_FLOAT, 
                                tensorflow::TensorShape({batch_size, height, width, channels}));
        auto images_flat = images.flat<float>();
        
        // Fill with fuzz data or zeros if not enough data
        for (int i = 0; i < images_flat.size(); ++i) {
            if (offset + 4 <= size) {
                images_flat(i) = *reinterpret_cast<const float*>(data + offset);
                offset += 4;
            } else {
                images_flat(i) = 0.0f;
            }
        }
        
        // Create scale tensor
        tensorflow::Tensor scale(tensorflow::DT_FLOAT, tensorflow::TensorShape({2}));
        scale.flat<float>()(0) = scale_x;
        scale.flat<float>()(1) = scale_y;
        
        // Create translation tensor
        tensorflow::Tensor translation(tensorflow::DT_FLOAT, tensorflow::TensorShape({2}));
        translation.flat<float>()(0) = translate_x;
        translation.flat<float>()(1) = translate_y;
        
        // Create session and graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node_def = graph_def.add_node();
        
        node_def->set_name("scale_and_translate");
        node_def->set_op("ScaleAndTranslate");
        node_def->add_input("images");
        node_def->add_input("size");
        node_def->add_input("scale");
        node_def->add_input("translation");
        
        // Set attributes
        tensorflow::AttrValue kernel_type_attr;
        kernel_type_attr.set_s(antialias ? "lanczos3" : "bilinear");
        (*node_def->mutable_attr())["kernel_type"] = kernel_type_attr;
        
        tensorflow::AttrValue antialias_attr;
        antialias_attr.set_b(antialias);
        (*node_def->mutable_attr())["antialias"] = antialias_attr;
        
        // Add input nodes
        tensorflow::NodeDef* images_node = graph_def.add_node();
        images_node->set_name("images");
        images_node->set_op("Placeholder");
        tensorflow::AttrValue dtype_attr;
        dtype_attr.set_type(tensorflow::DT_FLOAT);
        (*images_node->mutable_attr())["dtype"] = dtype_attr;
        
        tensorflow::NodeDef* size_node = graph_def.add_node();
        size_node->set_name("size");
        size_node->set_op("Placeholder");
        tensorflow::AttrValue size_dtype_attr;
        size_dtype_attr.set_type(tensorflow::DT_INT32);
        (*size_node->mutable_attr())["dtype"] = size_dtype_attr;
        
        tensorflow::NodeDef* scale_node = graph_def.add_node();
        scale_node->set_name("scale");
        scale_node->set_op("Placeholder");
        tensorflow::AttrValue scale_dtype_attr;
        scale_dtype_attr.set_type(tensorflow::DT_FLOAT);
        (*scale_node->mutable_attr())["dtype"] = scale_dtype_attr;
        
        tensorflow::NodeDef* translation_node = graph_def.add_node();
        translation_node->set_name("translation");
        translation_node->set_op("Placeholder");
        tensorflow::AttrValue translation_dtype_attr;
        translation_dtype_attr.set_type(tensorflow::DT_FLOAT);
        (*translation_node->mutable_attr())["dtype"] = translation_dtype_attr;
        
        // Create output size tensor
        int out_height = std::max(1, static_cast<int>(height * scale_y));
        int out_width = std::max(1, static_cast<int>(width * scale_x));
        tensorflow::Tensor size_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({2}));
        size_tensor.flat<int32_t>()(0) = out_height;
        size_tensor.flat<int32_t>()(1) = out_width;
        
        // Create session
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        if (!session) return 0;
        
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) return 0;
        
        // Run the operation
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"images", images},
            {"size", size_tensor},
            {"scale", scale},
            {"translation", translation}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"scale_and_translate"}, {}, &outputs);
        
        // Clean up
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}