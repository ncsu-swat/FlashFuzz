#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <vector>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create meaningful tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create a vector of parameters (tensors) with gradients
        std::vector<torch::Tensor> parameters;
        
        // Determine number of parameters to create (1-4)
        uint8_t num_params = (Size > 0) ? (Data[offset] % 4) + 1 : 1;
        offset++;
        
        // Create parameters with gradients
        for (uint8_t i = 0; i < num_params && offset < Size; i++) {
            // Create a tensor with random data
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Make it require gradients
            tensor = tensor.detach().requires_grad_(true);
            
            // Create a fake gradient for the tensor
            torch::Tensor grad;
            if (offset < Size) {
                grad = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Ensure gradient has same shape as tensor
                if (grad.sizes() != tensor.sizes()) {
                    grad = grad.reshape_as(tensor);
                }
                
                // Set the gradient
                tensor.mutable_grad() = grad;
            } else {
                // If we don't have enough data, create a simple gradient
                tensor.mutable_grad() = torch::ones_like(tensor);
            }
            
            parameters.push_back(tensor);
        }
        
        // Skip if no parameters with gradients
        if (parameters.empty()) {
            return 0;
        }
        
        // Extract max_norm from the input data
        double max_norm = 1.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&max_norm, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Extract norm_type from the input data
        double norm_type = 2.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&norm_type, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Extract error_if_nonfinite flag
        bool error_if_nonfinite = false;
        if (offset < Size) {
            error_if_nonfinite = Data[offset] % 2 == 1;
            offset++;
        }
        
        // Call clip_grad_norm_
        torch::nn::utils::clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite);
        
        // Try with a single tensor
        if (!parameters.empty()) {
            torch::nn::utils::clip_grad_norm_(parameters[0], max_norm, norm_type, error_if_nonfinite);
        }
        
        // Try with different max_norm values
        if (offset < Size) {
            max_norm = static_cast<double>(Data[offset]);
            torch::nn::utils::clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite);
            offset++;
        }
        
        // Try with different norm_type values
        if (offset < Size) {
            norm_type = static_cast<double>(Data[offset] % 10);  // Try different norm types (0-9)
            torch::nn::utils::clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite);
            offset++;
        }
        
        // Try with error_if_nonfinite = true
        torch::nn::utils::clip_grad_norm_(parameters, max_norm, norm_type, true);
        
        // Try with empty parameter list
        std::vector<torch::Tensor> empty_params;
        torch::nn::utils::clip_grad_norm_(empty_params, max_norm, norm_type, error_if_nonfinite);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}