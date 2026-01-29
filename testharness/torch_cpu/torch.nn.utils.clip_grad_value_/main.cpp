#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor and clip value
        if (Size < 8) {
            return 0;
        }
        
        // Create a tensor with gradient
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Set requires_grad to true to enable gradient tracking
        tensor = tensor.detach().requires_grad_(true);
        
        // Create a simple operation to generate gradients
        torch::Tensor output = tensor.pow(2).sum();
        
        // Backpropagate to compute gradients
        output.backward();
        
        // Extract clip value from the remaining data
        float clip_value = 1.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&clip_value, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Handle NaN/Inf and ensure positive value
        if (!std::isfinite(clip_value) || clip_value <= 0.0f) {
            clip_value = 1.0f;
        } else {
            clip_value = std::abs(clip_value);
        }
        
        // Create a vector of parameters to clip
        std::vector<torch::Tensor> parameters = {tensor};
        
        // Apply clip_grad_value_
        torch::nn::utils::clip_grad_value_(parameters, clip_value);
        
        // Test with a second tensor if we have enough data
        if (offset + 4 < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            tensor2 = tensor2.detach().requires_grad_(true);
            
            // Create computation graph and backward for tensor2
            torch::Tensor output2 = tensor2.pow(2).sum();
            output2.backward();
            
            // Extract new clip value
            float clip_value2 = 1.0f;
            if (offset + sizeof(float) <= Size) {
                std::memcpy(&clip_value2, Data + offset, sizeof(float));
                offset += sizeof(float);
                if (!std::isfinite(clip_value2) || clip_value2 <= 0.0f) {
                    clip_value2 = 1.0f;
                } else {
                    clip_value2 = std::abs(clip_value2);
                }
            }
            
            // Create a vector with both tensors (both have gradients now)
            std::vector<torch::Tensor> multiple_parameters = {tensor, tensor2};
            
            // Apply clip_grad_value_ to multiple tensors
            torch::nn::utils::clip_grad_value_(multiple_parameters, clip_value2);
        }
        
        // Test with very small clip value
        {
            torch::Tensor small_tensor = torch::randn({3, 3}, torch::requires_grad());
            torch::Tensor small_out = small_tensor.pow(2).sum();
            small_out.backward();
            
            std::vector<torch::Tensor> small_params = {small_tensor};
            float small_clip = 1e-6f;
            torch::nn::utils::clip_grad_value_(small_params, small_clip);
        }
        
        // Test with very large clip value
        {
            torch::Tensor large_tensor = torch::randn({3, 3}, torch::requires_grad());
            torch::Tensor large_out = large_tensor.pow(2).sum();
            large_out.backward();
            
            std::vector<torch::Tensor> large_params = {large_tensor};
            float large_clip = 1e6f;
            torch::nn::utils::clip_grad_value_(large_params, large_clip);
        }
        
        // Test with tensor that has no gradient (should be skipped gracefully)
        {
            torch::Tensor no_grad_tensor = torch::randn({2, 2});
            std::vector<torch::Tensor> no_grad_params = {no_grad_tensor};
            torch::nn::utils::clip_grad_value_(no_grad_params, 1.0f);
        }
        
        // Test with empty vector of parameters
        {
            std::vector<torch::Tensor> empty_params;
            torch::nn::utils::clip_grad_value_(empty_params, 1.0f);
        }
        
        // Test with tensor that has gradient but it's undefined
        {
            torch::Tensor undefined_grad_tensor = torch::randn({2, 2}, torch::requires_grad());
            // Don't call backward, so grad is undefined
            std::vector<torch::Tensor> undefined_grad_params = {undefined_grad_tensor};
            torch::nn::utils::clip_grad_value_(undefined_grad_params, 1.0f);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}