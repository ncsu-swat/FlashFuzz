#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
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
        float clip_value = 1.0f; // Default value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&clip_value, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Absolute value of clip_value to test both positive and negative values
        clip_value = std::abs(clip_value);
        
        // Create a vector of parameters to clip
        std::vector<torch::Tensor> parameters = {tensor};
        
        // Apply clip_grad_value_
        torch::nn::utils::clip_grad_value_(parameters, clip_value);
        
        // Verify that gradients are clipped
        if (tensor.grad().defined()) {
            // Check if all gradient values are within the clip range
            auto max_grad = torch::max(torch::abs(tensor.grad())).item<float>();
            if (max_grad > clip_value * 1.001f) { // Allow small floating point error
                throw std::runtime_error("Gradient clipping failed: max gradient exceeds clip value");
            }
        }
        
        // Try with a very small clip value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&clip_value, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Use a very small positive value to test edge cases
            clip_value = std::abs(clip_value) * 1e-10f + 1e-10f;
            
            // Reset gradients and recompute
            tensor.grad().zero_();
            output = tensor.pow(2).sum();
            output.backward();
            
            // Apply clip_grad_value_ with small value
            torch::nn::utils::clip_grad_value_(parameters, clip_value);
        }
        
        // Try with a very large clip value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&clip_value, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Use a very large value to test edge cases
            clip_value = std::abs(clip_value) * 1e10f + 1e10f;
            
            // Reset gradients and recompute
            tensor.grad().zero_();
            output = tensor.pow(2).sum();
            output.backward();
            
            // Apply clip_grad_value_ with large value
            torch::nn::utils::clip_grad_value_(parameters, clip_value);
        }
        
        // Try with multiple tensors of different shapes
        if (offset < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            tensor2 = tensor2.detach().requires_grad_(true);
            
            // Create a simple operation to generate gradients
            torch::Tensor output2 = tensor2.pow(2).sum();
            output2.backward();
            
            // Create a vector with both tensors
            std::vector<torch::Tensor> multiple_parameters = {tensor, tensor2};
            
            // Extract new clip value
            if (offset + sizeof(float) <= Size) {
                std::memcpy(&clip_value, Data + offset, sizeof(float));
                offset += sizeof(float);
                clip_value = std::abs(clip_value);
            }
            
            // Apply clip_grad_value_ to multiple tensors
            torch::nn::utils::clip_grad_value_(multiple_parameters, clip_value);
        }
        
        // Try with a tensor that has no gradient
        if (offset < Size) {
            torch::Tensor no_grad_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Don't call requires_grad_(true) or backward()
            
            // Create a vector with the no-grad tensor
            std::vector<torch::Tensor> no_grad_parameters = {no_grad_tensor};
            
            // Extract new clip value
            if (offset + sizeof(float) <= Size) {
                std::memcpy(&clip_value, Data + offset, sizeof(float));
                offset += sizeof(float);
                clip_value = std::abs(clip_value);
            }
            
            // Apply clip_grad_value_ to tensor with no gradient
            torch::nn::utils::clip_grad_value_(no_grad_parameters, clip_value);
        }
        
        // Try with an empty vector of parameters
        std::vector<torch::Tensor> empty_parameters;
        torch::nn::utils::clip_grad_value_(empty_parameters, clip_value);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
