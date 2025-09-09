#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract dimensions from fuzz data
        uint32_t num_boxes = (data[offset] % 8) + 1;
        offset += 1;
        uint32_t crop_height = (data[offset] % 16) + 1;
        offset += 1;
        uint32_t crop_width = (data[offset] % 16) + 1;
        offset += 1;
        uint32_t depth = (data[offset] % 8) + 1;
        offset += 1;
        uint32_t batch = (data[offset] % 4) + 1;
        offset += 1;
        uint32_t image_height = (data[offset] % 32) + 1;
        offset += 1;
        uint32_t image_width = (data[offset] % 32) + 1;
        offset += 1;
        
        // Calculate required data size
        size_t grads_size = num_boxes * crop_height * crop_width * depth * sizeof(float);
        size_t image_size = batch * image_height * image_width * depth * sizeof(float);
        size_t boxes_size = num_boxes * 4 * sizeof(float);
        size_t box_ind_size = num_boxes * sizeof(int32_t);
        
        size_t total_required = offset + grads_size + image_size + boxes_size + box_ind_size;
        if (size < total_required) return 0;
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create grads tensor
        tensorflow::Tensor grads_tensor(tensorflow::DT_FLOAT, 
            tensorflow::TensorShape({static_cast<int64_t>(num_boxes), 
                                    static_cast<int64_t>(crop_height),
                                    static_cast<int64_t>(crop_width), 
                                    static_cast<int64_t>(depth)}));
        
        auto grads_flat = grads_tensor.flat<float>();
        std::memcpy(grads_flat.data(), data + offset, grads_size);
        offset += grads_size;
        
        // Create image tensor
        tensorflow::Tensor image_tensor(tensorflow::DT_FLOAT,
            tensorflow::TensorShape({static_cast<int64_t>(batch),
                                    static_cast<int64_t>(image_height),
                                    static_cast<int64_t>(image_width),
                                    static_cast<int64_t>(depth)}));
        
        auto image_flat = image_tensor.flat<float>();
        std::memcpy(image_flat.data(), data + offset, image_size);
        offset += image_size;
        
        // Create boxes tensor
        tensorflow::Tensor boxes_tensor(tensorflow::DT_FLOAT,
            tensorflow::TensorShape({static_cast<int64_t>(num_boxes), 4}));
        
        auto boxes_flat = boxes_tensor.flat<float>();
        std::memcpy(boxes_flat.data(), data + offset, boxes_size);
        
        // Normalize box coordinates to [0, 1] range
        for (int i = 0; i < num_boxes * 4; i++) {
            float val = boxes_flat(i);
            if (std::isnan(val) || std::isinf(val)) {
                boxes_flat(i) = 0.5f;
            } else {
                boxes_flat(i) = std::max(0.0f, std::min(1.0f, std::abs(val)));
            }
        }
        offset += boxes_size;
        
        // Create box_ind tensor
        tensorflow::Tensor box_ind_tensor(tensorflow::DT_INT32,
            tensorflow::TensorShape({static_cast<int64_t>(num_boxes)}));
        
        auto box_ind_flat = box_ind_tensor.flat<int32_t>();
        std::memcpy(box_ind_flat.data(), data + offset, box_ind_size);
        
        // Ensure box indices are valid
        for (int i = 0; i < num_boxes; i++) {
            box_ind_flat(i) = std::abs(box_ind_flat(i)) % batch;
        }
        
        // Create input placeholders
        auto grads_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto image_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto boxes_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto box_ind_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        
        // Create CropAndResizeGradBoxes operation
        auto crop_resize_grad_boxes = tensorflow::ops::CropAndResizeGradBoxes(
            root, grads_ph, image_ph, boxes_ph, box_ind_ph,
            tensorflow::ops::CropAndResizeGradBoxes::Method("bilinear"));
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run(
            {{grads_ph, grads_tensor},
             {image_ph, image_tensor},
             {boxes_ph, boxes_tensor},
             {box_ind_ph, box_ind_tensor}},
            {crop_resize_grad_boxes}, &outputs);
        
        if (!status.ok()) {
            std::cout << "TensorFlow operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output shape
        if (!outputs.empty()) {
            const auto& output_shape = outputs[0].shape();
            if (output_shape.dims() == 2 && 
                output_shape.dim_size(0) == num_boxes && 
                output_shape.dim_size(1) == 4) {
                // Output shape is correct
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}