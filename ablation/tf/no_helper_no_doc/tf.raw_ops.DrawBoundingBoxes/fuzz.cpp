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
        
        if (size < 16) return 0;
        
        // Extract dimensions from fuzz data
        int batch_size = (data[offset] % 4) + 1;
        offset++;
        int height = (data[offset] % 128) + 32;
        offset++;
        int width = (data[offset] % 128) + 32;
        offset++;
        int channels = (data[offset] % 3) + 1;
        offset++;
        
        int num_boxes = (data[offset] % 10) + 1;
        offset++;
        
        if (offset + batch_size * height * width * channels + batch_size * num_boxes * 4 > size) {
            return 0;
        }
        
        // Create images tensor [batch_size, height, width, channels]
        tensorflow::Tensor images(tensorflow::DT_FLOAT, 
                                tensorflow::TensorShape({batch_size, height, width, channels}));
        auto images_flat = images.flat<float>();
        
        for (int i = 0; i < batch_size * height * width * channels && offset < size; i++) {
            images_flat(i) = static_cast<float>(data[offset]) / 255.0f;
            offset++;
        }
        
        // Create boxes tensor [batch_size, num_boxes, 4]
        tensorflow::Tensor boxes(tensorflow::DT_FLOAT, 
                               tensorflow::TensorShape({batch_size, num_boxes, 4}));
        auto boxes_flat = boxes.flat<float>();
        
        for (int i = 0; i < batch_size * num_boxes * 4 && offset < size; i++) {
            boxes_flat(i) = static_cast<float>(data[offset]) / 255.0f;
            offset++;
        }
        
        // Create session and graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* images_node = graph_def.add_node();
        images_node->set_name("images");
        images_node->set_op("Placeholder");
        (*images_node->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        (*images_node->mutable_attr())["shape"].mutable_shape()->add_dim()->set_size(batch_size);
        (*images_node->mutable_attr())["shape"].mutable_shape()->add_dim()->set_size(height);
        (*images_node->mutable_attr())["shape"].mutable_shape()->add_dim()->set_size(width);
        (*images_node->mutable_attr())["shape"].mutable_shape()->add_dim()->set_size(channels);
        
        tensorflow::NodeDef* boxes_node = graph_def.add_node();
        boxes_node->set_name("boxes");
        boxes_node->set_op("Placeholder");
        (*boxes_node->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        (*boxes_node->mutable_attr())["shape"].mutable_shape()->add_dim()->set_size(batch_size);
        (*boxes_node->mutable_attr())["shape"].mutable_shape()->add_dim()->set_size(num_boxes);
        (*boxes_node->mutable_attr())["shape"].mutable_shape()->add_dim()->set_size(4);
        
        tensorflow::NodeDef* draw_node = graph_def.add_node();
        draw_node->set_name("draw_bounding_boxes");
        draw_node->set_op("DrawBoundingBoxes");
        draw_node->add_input("images");
        draw_node->add_input("boxes");
        (*draw_node->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
        
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({{"images", images}, {"boxes", boxes}}, 
                            {"draw_bounding_boxes"}, {}, &outputs);
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}