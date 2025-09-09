#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/kernels/crop_and_resize_op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/framework/node_def_util.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract dimensions from fuzz data
        int32_t batch_size = (data[offset] % 8) + 1;
        offset++;
        int32_t image_height = (data[offset] % 64) + 8;
        offset++;
        int32_t image_width = (data[offset] % 64) + 8;
        offset++;
        int32_t depth = (data[offset] % 8) + 1;
        offset++;
        int32_t num_boxes = (data[offset] % 16) + 1;
        offset++;
        int32_t crop_height = (data[offset] % 32) + 4;
        offset++;
        int32_t crop_width = (data[offset] % 32) + 4;
        offset++;
        
        // Create input tensors
        tensorflow::Tensor grads(tensorflow::DT_FLOAT, 
            tensorflow::TensorShape({num_boxes, crop_height, crop_width, depth}));
        
        tensorflow::Tensor images(tensorflow::DT_FLOAT,
            tensorflow::TensorShape({batch_size, image_height, image_width, depth}));
            
        tensorflow::Tensor boxes(tensorflow::DT_FLOAT,
            tensorflow::TensorShape({num_boxes, 4}));
            
        tensorflow::Tensor box_ind(tensorflow::DT_INT32,
            tensorflow::TensorShape({num_boxes}));
        
        // Fill tensors with fuzz data
        auto grads_flat = grads.flat<float>();
        auto images_flat = images.flat<float>();
        auto boxes_flat = boxes.flat<float>();
        auto box_ind_flat = box_ind.flat<int32_t>();
        
        size_t data_needed = grads_flat.size() * sizeof(float) + 
                           images_flat.size() * sizeof(float) +
                           boxes_flat.size() * sizeof(float) +
                           box_ind_flat.size() * sizeof(int32_t);
        
        if (offset + data_needed > size) return 0;
        
        // Fill grads tensor
        for (int i = 0; i < grads_flat.size() && offset + 4 <= size; i++) {
            float val;
            memcpy(&val, data + offset, sizeof(float));
            grads_flat(i) = val;
            offset += sizeof(float);
        }
        
        // Fill images tensor
        for (int i = 0; i < images_flat.size() && offset + 4 <= size; i++) {
            float val;
            memcpy(&val, data + offset, sizeof(float));
            images_flat(i) = val;
            offset += sizeof(float);
        }
        
        // Fill boxes tensor with normalized coordinates [0, 1]
        for (int i = 0; i < boxes_flat.size() && offset + 4 <= size; i++) {
            float val;
            memcpy(&val, data + offset, sizeof(float));
            // Normalize to [0, 1] range for valid box coordinates
            boxes_flat(i) = std::abs(val) - std::floor(std::abs(val));
            offset += sizeof(float);
        }
        
        // Fill box_ind tensor with valid batch indices
        for (int i = 0; i < box_ind_flat.size() && offset + 4 <= size; i++) {
            int32_t val;
            memcpy(&val, data + offset, sizeof(int32_t));
            box_ind_flat(i) = std::abs(val) % batch_size;
            offset += sizeof(int32_t);
        }
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create graph def
        tensorflow::GraphDef graph_def;
        tensorflow::GraphDefBuilder builder(tensorflow::GraphDefBuilder::kFailImmediately);
        
        // Add placeholders
        auto grads_ph = tensorflow::ops::Placeholder(builder.opts()
            .WithName("grads")
            .WithAttr("dtype", tensorflow::DT_FLOAT)
            .WithAttr("shape", grads.shape()));
            
        auto images_ph = tensorflow::ops::Placeholder(builder.opts()
            .WithName("images") 
            .WithAttr("dtype", tensorflow::DT_FLOAT)
            .WithAttr("shape", images.shape()));
            
        auto boxes_ph = tensorflow::ops::Placeholder(builder.opts()
            .WithName("boxes")
            .WithAttr("dtype", tensorflow::DT_FLOAT)
            .WithAttr("shape", boxes.shape()));
            
        auto box_ind_ph = tensorflow::ops::Placeholder(builder.opts()
            .WithName("box_ind")
            .WithAttr("dtype", tensorflow::DT_INT32)
            .WithAttr("shape", box_ind.shape()));
        
        // Add CropAndResizeGradBoxes operation
        tensorflow::NodeDef node_def;
        node_def.set_name("crop_and_resize_grad_boxes");
        node_def.set_op("CropAndResizeGradBoxes");
        node_def.add_input("grads");
        node_def.add_input("images");
        node_def.add_input("boxes");
        node_def.add_input("box_ind");
        (*node_def.mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
        (*node_def.mutable_attr())["method"].set_s("bilinear");
        
        tensorflow::Status status = builder.ToGraphDef(&graph_def);
        if (!status.ok()) return 0;
        
        // Add the CropAndResizeGradBoxes node manually
        *graph_def.add_node() = node_def;
        
        status = session->Create(graph_def);
        if (!status.ok()) return 0;
        
        // Run the operation
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"grads", grads},
            {"images", images}, 
            {"boxes", boxes},
            {"box_ind", box_ind}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"crop_and_resize_grad_boxes"}, {}, &outputs);
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}