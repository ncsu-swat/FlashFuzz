#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for input tensor and parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create PReLU module with different configurations
        if (offset < Size) {
            // Use remaining data to determine PReLU configuration
            uint8_t config_byte = Data[offset++];
            
            // Determine if we should use a single parameter or per-channel parameters
            bool use_per_channel = (config_byte & 0x01);
            
            // Create PReLU module
            torch::nn::PReLU prelu;
            
            if (use_per_channel && input.dim() > 1) {
                // For per-channel PReLU, we need to specify the number of parameters
                // which should match the number of channels (dimension 1)
                int64_t num_params = input.size(1);
                
                // Initialize with random values between 0 and 1
                torch::Tensor weight = torch::rand(num_params);
                
                // Set the weight parameter
                prelu->weight = weight;
            } else {
                // Single parameter PReLU
                // Use a single value from the input data if available
                float param_value = 0.25; // Default value
                if (offset + sizeof(float) <= Size) {
                    std::memcpy(&param_value, Data + offset, sizeof(float));
                    offset += sizeof(float);
                }
                
                // Set the weight parameter
                prelu->weight = torch::tensor({param_value});
            }
            
            // Apply PReLU to the input tensor
            torch::Tensor output = prelu->forward(input);
            
            // Verify the output has the same shape as the input
            if (output.sizes() != input.sizes()) {
                throw std::runtime_error("PReLU output shape doesn't match input shape");
            }
            
            // Try another approach: use the functional interface
            torch::Tensor output2 = torch::prelu(input, prelu->weight);
            
            // Verify both approaches give the same result
            if (!torch::allclose(output, output2)) {
                throw std::runtime_error("PReLU module and functional interface give different results");
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}