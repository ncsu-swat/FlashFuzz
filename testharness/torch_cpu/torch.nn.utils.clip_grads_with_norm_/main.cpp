#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create a list of tensors to represent parameters with gradients
        std::vector<torch::Tensor> parameters;
        
        // Determine number of tensors to create (1-4 based on available data)
        uint8_t num_tensors = 1;
        if (offset < Size) {
            num_tensors = (Data[offset++] % 4) + 1;
        }
        
        // Create tensors with gradients
        for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
            // Create a tensor
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Make it require gradients
            tensor = tensor.detach().requires_grad_(true);
            
            // Create a gradient for the tensor
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
            
            // Add to parameters list
            parameters.push_back(tensor);
        }
        
        // Skip if no parameters with gradients
        if (parameters.empty()) {
            return 0;
        }
        
        // Parse max_norm parameter
        double max_norm = 1.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&max_norm, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Parse norm_type parameter
        double norm_type = 2.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&norm_type, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Apply clip_grad_norm_ operation
        torch::nn::utils::clip_grad_norm_(parameters, max_norm, norm_type);
        
        // Optionally test clip_grad_value_ as well if we have more data
        if (offset < Size) {
            double clip_value = 1.0;
            std::memcpy(&clip_value, Data + offset, sizeof(double));
            torch::nn::utils::clip_grad_value_(parameters, clip_value);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}