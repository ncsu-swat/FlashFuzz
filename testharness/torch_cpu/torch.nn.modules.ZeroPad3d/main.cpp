#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Need at least some data to proceed
        if (Size < 10) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract padding values first (use bytes, bounded to reasonable range)
        // Use int8_t to get values in range [-128, 127], then bound further
        int64_t padding_left = static_cast<int8_t>(Data[offset++]) % 16;
        int64_t padding_right = static_cast<int8_t>(Data[offset++]) % 16;
        int64_t padding_top = static_cast<int8_t>(Data[offset++]) % 16;
        int64_t padding_bottom = static_cast<int8_t>(Data[offset++]) % 16;
        int64_t padding_front = static_cast<int8_t>(Data[offset++]) % 16;
        int64_t padding_back = static_cast<int8_t>(Data[offset++]) % 16;
        
        // Create input tensor from remaining data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
        
        // Ensure we have a 5D tensor (batch, channels, depth, height, width)
        // ZeroPad3d also accepts 4D tensors (channels, depth, height, width)
        if (input_tensor.dim() < 4) {
            std::vector<int64_t> new_shape;
            int64_t numel = input_tensor.numel();
            
            // Create a small 4D or 5D tensor
            if (numel > 0) {
                new_shape = {1, 1, 1, 1, numel};
                input_tensor = input_tensor.reshape(new_shape);
            } else {
                return 0;
            }
        } else if (input_tensor.dim() == 4) {
            // 4D is valid for ZeroPad3d, keep as is
        } else if (input_tensor.dim() > 5) {
            // Flatten extra dimensions into batch
            int64_t batch_size = 1;
            for (int i = 0; i < input_tensor.dim() - 4; i++) {
                batch_size *= input_tensor.size(i);
            }
            std::vector<int64_t> new_shape = {batch_size};
            for (int i = input_tensor.dim() - 4; i < input_tensor.dim(); i++) {
                new_shape.push_back(input_tensor.size(i));
            }
            input_tensor = input_tensor.reshape(new_shape);
        }
        
        // Case 1: Single integer for all sides
        try {
            int64_t single_padding = std::abs(padding_left) % 8;
            auto pad_module1 = torch::nn::ZeroPad3d(single_padding);
            auto output1 = pad_module1->forward(input_tensor);
            (void)output1;
        } catch (const std::exception &) {
            // Expected for some configurations
        }
        
        // Case 2: Tuple of 6 values (left, right, top, bottom, front, back)
        try {
            auto pad_module2 = torch::nn::ZeroPad3d(torch::nn::ZeroPad3dOptions(
                {std::abs(padding_left) % 8, std::abs(padding_right) % 8, 
                 std::abs(padding_top) % 8, std::abs(padding_bottom) % 8, 
                 std::abs(padding_front) % 8, std::abs(padding_back) % 8}));
            auto output2 = pad_module2->forward(input_tensor);
            (void)output2;
        } catch (const std::exception &) {
            // Expected for some configurations
        }
        
        // Case 3: Using functional interface
        try {
            auto output3 = torch::nn::functional::pad(
                input_tensor,
                torch::nn::functional::PadFuncOptions(
                    {std::abs(padding_front) % 8, std::abs(padding_back) % 8, 
                     std::abs(padding_top) % 8, std::abs(padding_bottom) % 8, 
                     std::abs(padding_left) % 8, std::abs(padding_right) % 8})
                    .mode(torch::kConstant)
                    .value(0.0));
            (void)output3;
        } catch (const std::exception &) {
            // Expected for some configurations
        }
        
        // Case 4: Try with negative padding (cropping) - only if tensor is large enough
        try {
            int64_t neg_pad = -(std::abs(padding_left) % 4 + 1);
            // Only apply negative padding if tensor dimensions are large enough
            if (input_tensor.size(-1) > std::abs(neg_pad) * 2 &&
                input_tensor.size(-2) > std::abs(neg_pad) * 2 &&
                input_tensor.size(-3) > std::abs(neg_pad) * 2) {
                auto pad_module4 = torch::nn::ZeroPad3d(torch::nn::ZeroPad3dOptions(
                    {neg_pad, neg_pad, neg_pad, neg_pad, neg_pad, neg_pad}));
                auto output4 = pad_module4->forward(input_tensor);
                (void)output4;
            }
        } catch (const std::exception &) {
            // Expected for some configurations
        }
        
        // Case 5: Asymmetric padding
        try {
            auto pad_module5 = torch::nn::ZeroPad3d(torch::nn::ZeroPad3dOptions(
                {padding_left % 5, padding_right % 5, 
                 padding_top % 5, padding_bottom % 5, 
                 padding_front % 5, padding_back % 5}));
            auto output5 = pad_module5->forward(input_tensor);
            (void)output5;
        } catch (const std::exception &) {
            // Expected for some configurations
        }
        
        // Case 6: Zero padding (no-op)
        try {
            auto pad_module6 = torch::nn::ZeroPad3d(0);
            auto output6 = pad_module6->forward(input_tensor);
            (void)output6;
        } catch (const std::exception &) {
            // Expected for some configurations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}