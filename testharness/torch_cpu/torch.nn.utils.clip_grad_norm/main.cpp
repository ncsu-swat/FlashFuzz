#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create a vector of parameters (tensors with gradients)
        std::vector<torch::Tensor> parameters;
        
        // Determine number of tensors to create (1-4)
        uint8_t num_tensors = (Size > 0) ? (Data[offset++] % 4) + 1 : 1;
        
        // Create tensors with gradients
        for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
            // Create a tensor with requires_grad=true
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            tensor = tensor.detach().requires_grad_(true);
            
            // Create a "gradient" tensor with the same shape
            torch::Tensor grad;
            if (offset < Size) {
                grad = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Ensure gradient has the same shape as the tensor
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
        
        // Parse max_norm parameter (can be any float, including negative or zero)
        float max_norm = 1.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&max_norm, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Parse norm_type parameter (can be any float, including 0, 1, 2, inf, -inf)
        float norm_type = 2.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&norm_type, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Parse error_if_nonfinite flag
        bool error_if_nonfinite = false;
        if (offset < Size) {
            error_if_nonfinite = Data[offset++] & 0x1;
        }
        
        // Call clip_grad_norm_
        double total_norm = torch::nn::utils::clip_grad_norm_(
            parameters, 
            max_norm,
            norm_type,
            error_if_nonfinite
        );
        
        // Verify the result is a valid double
        if (std::isnan(total_norm) && !error_if_nonfinite) {
            // This is expected behavior in some cases
        }
        
        // Test the in-place version with a single tensor
        if (!parameters.empty()) {
            torch::Tensor single_tensor = parameters[0];
            double single_norm = torch::nn::utils::clip_grad_norm_(
                single_tensor, 
                max_norm,
                norm_type,
                error_if_nonfinite
            );
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}