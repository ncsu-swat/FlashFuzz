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
        
        // Extract dimensions from fuzz data
        int batch_size = (data[offset] % 4) + 1;
        offset++;
        int num_classes = (data[offset] % 8) + 1;
        offset++;
        int num_boxes = (data[offset] % 16) + 1;
        offset++;
        int max_output_size_per_class = (data[offset] % 8) + 1;
        offset++;
        int max_total_size = (data[offset] % 16) + 1;
        offset++;
        
        // Extract scalar parameters
        float iou_threshold = 0.5f;
        float score_threshold = 0.1f;
        if (offset + 8 < size) {
            memcpy(&iou_threshold, data + offset, sizeof(float));
            offset += sizeof(float);
            memcpy(&score_threshold, data + offset, sizeof(float));
            offset += sizeof(float);
            // Clamp values to reasonable ranges
            iou_threshold = std::max(0.0f, std::min(1.0f, iou_threshold));
            score_threshold = std::max(0.0f, std::min(1.0f, score_threshold));
        }
        
        // Create input tensors
        tensorflow::Tensor boxes(tensorflow::DT_FLOAT, 
            tensorflow::TensorShape({batch_size, num_boxes, 4}));
        tensorflow::Tensor scores(tensorflow::DT_FLOAT,
            tensorflow::TensorShape({batch_size, num_boxes, num_classes}));
        tensorflow::Tensor max_output_size_per_class_tensor(tensorflow::DT_INT32,
            tensorflow::TensorShape({}));
        tensorflow::Tensor max_total_size_tensor(tensorflow::DT_INT32,
            tensorflow::TensorShape({}));
        tensorflow::Tensor iou_threshold_tensor(tensorflow::DT_FLOAT,
            tensorflow::TensorShape({}));
        tensorflow::Tensor score_threshold_tensor(tensorflow::DT_FLOAT,
            tensorflow::TensorShape({}));
        
        // Fill tensors with fuzz data
        auto boxes_flat = boxes.flat<float>();
        auto scores_flat = scores.flat<float>();
        
        size_t boxes_size = batch_size * num_boxes * 4;
        size_t scores_size = batch_size * num_boxes * num_classes;
        
        for (int i = 0; i < boxes_size && offset < size; i++) {
            float val = static_cast<float>(data[offset % size]) / 255.0f;
            boxes_flat(i) = val;
            offset++;
        }
        
        for (int i = 0; i < scores_size && offset < size; i++) {
            float val = static_cast<float>(data[offset % size]) / 255.0f;
            scores_flat(i) = val;
            offset++;
        }
        
        // Set scalar values
        max_output_size_per_class_tensor.scalar<int32_t>()() = max_output_size_per_class;
        max_total_size_tensor.scalar<int32_t>()() = max_total_size;
        iou_threshold_tensor.scalar<float>()() = iou_threshold;
        score_threshold_tensor.scalar<float>()() = score_threshold;
        
        // Create session and graph
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node = graph_def.add_node();
        node->set_name("combined_nms");
        node->set_op("CombinedNonMaxSuppression");
        
        // Add input names
        node->add_input("boxes");
        node->add_input("scores");
        node->add_input("max_output_size_per_class");
        node->add_input("max_total_size");
        node->add_input("iou_threshold");
        node->add_input("score_threshold");
        
        // Add placeholder nodes for inputs
        auto add_placeholder = [&](const std::string& name, tensorflow::DataType dtype, 
                                  const tensorflow::TensorShape& shape) {
            tensorflow::NodeDef* placeholder = graph_def.add_node();
            placeholder->set_name(name);
            placeholder->set_op("Placeholder");
            (*placeholder->mutable_attr())["dtype"].set_type(dtype);
            (*placeholder->mutable_attr())["shape"].mutable_shape()->CopyFrom(shape.AsProto());
        };
        
        add_placeholder("boxes", tensorflow::DT_FLOAT, boxes.shape());
        add_placeholder("scores", tensorflow::DT_FLOAT, scores.shape());
        add_placeholder("max_output_size_per_class", tensorflow::DT_INT32, 
                       max_output_size_per_class_tensor.shape());
        add_placeholder("max_total_size", tensorflow::DT_INT32, max_total_size_tensor.shape());
        add_placeholder("iou_threshold", tensorflow::DT_FLOAT, iou_threshold_tensor.shape());
        add_placeholder("score_threshold", tensorflow::DT_FLOAT, score_threshold_tensor.shape());
        
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Prepare inputs
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"boxes", boxes},
            {"scores", scores},
            {"max_output_size_per_class", max_output_size_per_class_tensor},
            {"max_total_size", max_total_size_tensor},
            {"iou_threshold", iou_threshold_tensor},
            {"score_threshold", score_threshold_tensor}
        };
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {
            "combined_nms:0", "combined_nms:1", "combined_nms:2", "combined_nms:3"
        };
        
        status = session->Run(inputs, output_names, {}, &outputs);
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}