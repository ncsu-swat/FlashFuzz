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
        
        // Extract dimensions for images tensor
        uint32_t batch = (data[offset] % 4) + 1;
        offset++;
        uint32_t height = (data[offset] % 64) + 1;
        offset++;
        uint32_t width = (data[offset] % 64) + 1;
        offset++;
        uint32_t channels = (data[offset] % 4) + 1;
        offset++;
        
        // Extract new size dimensions
        uint32_t new_height = (data[offset] % 64) + 1;
        offset++;
        uint32_t new_width = (data[offset] % 64) + 1;
        offset++;
        
        // Extract scale values
        float scale_x = (data[offset] % 100) / 50.0f + 0.1f;
        offset++;
        float scale_y = (data[offset] % 100) / 50.0f + 0.1f;
        offset++;
        
        // Extract translation values
        float trans_x = (data[offset] % 200) - 100.0f;
        offset++;
        float trans_y = (data[offset] % 200) - 100.0f;
        offset++;
        
        // Extract kernel type and antialias
        uint8_t kernel_idx = data[offset] % 4;
        offset++;
        bool antialias = (data[offset] % 2) == 1;
        offset++;
        
        std::string kernel_types[] = {"lanczos3", "lanczos5", "gaussian", "box"};
        std::string kernel_type = kernel_types[kernel_idx];
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create images tensor
        tensorflow::Tensor images_tensor(tensorflow::DT_FLOAT, 
            tensorflow::TensorShape({batch, height, width, channels}));
        auto images_flat = images_tensor.flat<float>();
        
        // Fill images tensor with fuzz data
        size_t tensor_size = batch * height * width * channels;
        for (size_t i = 0; i < tensor_size && offset < size; i++, offset++) {
            images_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
        }
        
        // Create size tensor
        tensorflow::Tensor size_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({2}));
        auto size_flat = size_tensor.flat<int32_t>();
        size_flat(0) = new_height;
        size_flat(1) = new_width;
        
        // Create scale tensor
        tensorflow::Tensor scale_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({2}));
        auto scale_flat = scale_tensor.flat<float>();
        scale_flat(0) = scale_y;
        scale_flat(1) = scale_x;
        
        // Create translation tensor
        tensorflow::Tensor translation_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({2}));
        auto translation_flat = translation_tensor.flat<float>();
        translation_flat(0) = trans_y;
        translation_flat(1) = trans_x;
        
        // Create input placeholders
        auto images_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto size_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto scale_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto translation_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        // Create ScaleAndTranslate operation
        auto scale_and_translate = tensorflow::ops::ScaleAndTranslate(
            root, images_ph, size_ph, scale_ph, translation_ph,
            tensorflow::ops::ScaleAndTranslate::KernelType(kernel_type)
                .Antialias(antialias));
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run(
            {{images_ph, images_tensor}, 
             {size_ph, size_tensor},
             {scale_ph, scale_tensor},
             {translation_ph, translation_tensor}},
            {scale_and_translate}, &outputs);
        
        if (!status.ok()) {
            std::cout << "TensorFlow operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output
        if (!outputs.empty() && outputs[0].dtype() == tensorflow::DT_FLOAT) {
            auto output_shape = outputs[0].shape();
            if (output_shape.dims() == 4 && 
                output_shape.dim_size(0) == batch &&
                output_shape.dim_size(1) == new_height &&
                output_shape.dim_size(2) == new_width &&
                output_shape.dim_size(3) == channels) {
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