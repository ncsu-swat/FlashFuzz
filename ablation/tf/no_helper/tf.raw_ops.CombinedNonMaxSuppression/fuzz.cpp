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
        uint32_t batch_size = (data[offset] % 4) + 1;
        offset += 1;
        uint32_t num_boxes = (data[offset] % 10) + 1;
        offset += 1;
        uint32_t num_classes = (data[offset] % 5) + 1;
        offset += 1;
        uint32_t q = (data[offset] % 2) == 0 ? 1 : num_classes;
        offset += 1;
        
        // Extract scalar parameters
        uint32_t max_output_size_per_class = (data[offset] % 10) + 1;
        offset += 1;
        uint32_t max_total_size = (data[offset] % 20) + 1;
        offset += 1;
        
        float iou_threshold = (data[offset] % 100) / 100.0f;
        offset += 1;
        float score_threshold = (data[offset] % 100) / 100.0f;
        offset += 1;
        
        bool pad_per_class = (data[offset] % 2) == 1;
        offset += 1;
        bool clip_boxes = (data[offset] % 2) == 1;
        offset += 1;
        
        // Create TensorFlow session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create input tensors
        tensorflow::Tensor boxes_tensor(tensorflow::DT_FLOAT, 
            tensorflow::TensorShape({static_cast<int64_t>(batch_size), 
                                    static_cast<int64_t>(num_boxes), 
                                    static_cast<int64_t>(q), 4}));
        
        tensorflow::Tensor scores_tensor(tensorflow::DT_FLOAT,
            tensorflow::TensorShape({static_cast<int64_t>(batch_size),
                                    static_cast<int64_t>(num_boxes),
                                    static_cast<int64_t>(num_classes)}));
        
        tensorflow::Tensor max_output_size_per_class_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        tensorflow::Tensor max_total_size_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        tensorflow::Tensor iou_threshold_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor score_threshold_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        
        // Fill tensors with fuzz data
        auto boxes_flat = boxes_tensor.flat<float>();
        auto scores_flat = scores_tensor.flat<float>();
        
        size_t boxes_size = batch_size * num_boxes * q * 4;
        size_t scores_size = batch_size * num_boxes * num_classes;
        
        for (size_t i = 0; i < boxes_size && offset < size; ++i, ++offset) {
            boxes_flat(i) = (data[offset] % 200) / 100.0f - 1.0f; // Range [-1, 1]
        }
        
        for (size_t i = 0; i < scores_size && offset < size; ++i, ++offset) {
            scores_flat(i) = (data[offset] % 100) / 100.0f; // Range [0, 1]
        }
        
        max_output_size_per_class_tensor.scalar<int32_t>()() = max_output_size_per_class;
        max_total_size_tensor.scalar<int32_t>()() = max_total_size;
        iou_threshold_tensor.scalar<float>()() = iou_threshold;
        score_threshold_tensor.scalar<float>()() = score_threshold;
        
        // Create graph definition
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node_def = graph_def.add_node();
        node_def->set_name("combined_nms");
        node_def->set_op("CombinedNonMaxSuppression");
        
        // Add input names
        node_def->add_input("boxes");
        node_def->add_input("scores");
        node_def->add_input("max_output_size_per_class");
        node_def->add_input("max_total_size");
        node_def->add_input("iou_threshold");
        node_def->add_input("score_threshold");
        
        // Set attributes
        (*node_def->mutable_attr())["pad_per_class"].set_b(pad_per_class);
        (*node_def->mutable_attr())["clip_boxes"].set_b(clip_boxes);
        
        // Add placeholder nodes for inputs
        auto add_placeholder = [&](const std::string& name, tensorflow::DataType dtype, const tensorflow::TensorShape& shape) {
            tensorflow::NodeDef* placeholder = graph_def.add_node();
            placeholder->set_name(name);
            placeholder->set_op("Placeholder");
            (*placeholder->mutable_attr())["dtype"].set_type(dtype);
            (*placeholder->mutable_attr())["shape"].mutable_shape()->CopyFrom(shape.AsProto());
        };
        
        add_placeholder("boxes", tensorflow::DT_FLOAT, boxes_tensor.shape());
        add_placeholder("scores", tensorflow::DT_FLOAT, scores_tensor.shape());
        add_placeholder("max_output_size_per_class", tensorflow::DT_INT32, tensorflow::TensorShape({}));
        add_placeholder("max_total_size", tensorflow::DT_INT32, tensorflow::TensorShape({}));
        add_placeholder("iou_threshold", tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        add_placeholder("score_threshold", tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        
        // Create session and run
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"boxes", boxes_tensor},
            {"scores", scores_tensor},
            {"max_output_size_per_class", max_output_size_per_class_tensor},
            {"max_total_size", max_total_size_tensor},
            {"iou_threshold", iou_threshold_tensor},
            {"score_threshold", score_threshold_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {
            "combined_nms:0", "combined_nms:1", "combined_nms:2", "combined_nms:3"
        };
        
        status = session->Run(inputs, output_names, {}, &outputs);
        
        if (status.ok() && outputs.size() == 4) {
            // Verify output shapes and types
            if (outputs[0].dtype() == tensorflow::DT_FLOAT &&
                outputs[1].dtype() == tensorflow::DT_FLOAT &&
                outputs[2].dtype() == tensorflow::DT_FLOAT &&
                outputs[3].dtype() == tensorflow::DT_INT32) {
                // Operation completed successfully
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