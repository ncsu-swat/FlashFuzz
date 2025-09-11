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
        
        // Get normalized_shape from the last dimension of the input tensor
        std::vector<int64_t> normalized_shape;
        if (input.dim() > 0) {
            int64_t last_dim = input.size(-1);
            normalized_shape.push_back(last_dim);
        } else {
            // For scalar tensors, use a default shape
            normalized_shape.push_back(1);
        }
        
        // Extract epsilon parameter from the input data
        double epsilon = 1e-5; // Default value
        if (offset + sizeof(float) <= Size) {
            float eps_raw;
            std::memcpy(&eps_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure epsilon is positive but allow very small values to test edge cases
            if (eps_raw > 0) {
                epsilon = eps_raw;
            } else if (eps_raw < 0) {
                epsilon = std::abs(eps_raw);
            }
        }
        
        // Apply RMSNorm using the functional API
        torch::Tensor output = torch::nn::functional::rms_norm(input, normalized_shape, torch::nullopt, epsilon);
        
        // Test with different weight configurations
        if (offset + 1 <= Size) {
            bool use_weight = Data[offset++] & 1;
            
            if (use_weight && input.dim() > 0) {
                // Create a weight tensor with the same shape as normalized_shape
                torch::Tensor weight = torch::ones(normalized_shape);
                
                // Apply RMSNorm with weight
                torch::Tensor output_with_weight = torch::nn::functional::rms_norm(input, normalized_shape, weight, epsilon);
            }
        }
        
        // Test with different data types if there's enough data left
        if (offset + 1 <= Size && input.dim() > 0) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Convert input to the new data type
            torch::Tensor input_converted = input.to(dtype);
            
            // Apply RMSNorm to the converted input
            torch::Tensor output_dtype = torch::nn::functional::rms_norm(input_converted, normalized_shape, torch::nullopt, epsilon);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
