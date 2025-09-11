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
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor has at least 2 dimensions for Dropout2d
        if (input.dim() < 2) {
            // Reshape to 2D if needed
            int64_t total_elements = input.numel();
            if (total_elements > 0) {
                input = input.reshape({1, total_elements});
            } else {
                input = torch::zeros({1, 1}, input.options());
            }
        }
        
        // Extract p (dropout probability) from input data
        float p = 0.5; // Default value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Clamp p to valid range [0, 1]
            p = std::abs(p);
            p = p - std::floor(p); // Get fractional part to ensure 0 <= p <= 1
        }
        
        // Extract inplace flag from input data
        bool inplace = false;
        if (offset < Size) {
            inplace = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Create Dropout2d module with parameters from fuzzer data
        torch::nn::Dropout2d dropout_module(
            torch::nn::Dropout2dOptions()
                .p(p)
                .inplace(inplace)
        );
        
        // Set training mode based on fuzzer data
        bool training_mode = true;
        if (offset < Size) {
            training_mode = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        if (training_mode) {
            dropout_module->train();
        } else {
            dropout_module->eval();
        }
        
        // Apply Dropout2d to the input tensor
        torch::Tensor output = dropout_module->forward(input);
        
        // Try to access output properties to ensure computation completed
        auto output_sizes = output.sizes();
        auto output_dtype = output.dtype();
        
        // Try some additional operations on the output
        if (output.numel() > 0) {
            auto sum = output.sum();
            auto mean = output.mean();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
