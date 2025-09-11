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
        
        // Need at least a few bytes for the fuzzer to work
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor with gradients
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Make the tensor require gradients
        tensor = tensor.detach().requires_grad_(true);
        
        // Create a simple operation to generate gradients
        torch::Tensor output = tensor.sum();
        
        // Backpropagate to compute gradients
        output.backward();
        
        // Extract max_norm parameter from the input data
        double max_norm = 0.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&max_norm, Data + offset, sizeof(double));
            offset += sizeof(double);
        } else {
            // Use a default value if not enough data
            max_norm = 1.0;
        }
        
        // Extract norm_type parameter from the input data
        double norm_type = 0.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&norm_type, Data + offset, sizeof(double));
            offset += sizeof(double);
        } else {
            // Use a default value if not enough data
            norm_type = 2.0;
        }
        
        // Create a vector of parameters
        std::vector<torch::Tensor> parameters;
        parameters.push_back(tensor);
        
        // Apply clip_grad_norm_
        torch::nn::utils::clip_grad_norm_(parameters, max_norm, norm_type);
        
        // Try clip_grad_value_ as well
        if (offset + sizeof(double) <= Size) {
            double clip_value;
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
