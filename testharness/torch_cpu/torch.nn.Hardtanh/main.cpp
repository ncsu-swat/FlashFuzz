#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for Hardtanh from the remaining data
        float min_val = -1.0;
        float max_val = 1.0;
        
        // If we have more data, use it for min_val and max_val
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&min_val, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&max_val, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Create Hardtanh module
        torch::nn::Hardtanh hardtanh(torch::nn::HardtanhOptions().min_val(min_val).max_val(max_val));
        
        // Apply Hardtanh to the input tensor
        torch::Tensor output = hardtanh->forward(input);
        
        // Try functional version as well
        torch::Tensor output_functional = torch::nn::functional::hardtanh(
            input, 
            torch::nn::functional::HardtanhFuncOptions().min_val(min_val).max_val(max_val)
        );
        
        // Try inplace version using hardtanh with inplace option
        torch::Tensor input_copy = input.clone();
        torch::nn::functional::hardtanh(
            input_copy,
            torch::nn::functional::HardtanhFuncOptions().min_val(min_val).max_val(max_val).inplace(true)
        );
        
        // Try with default parameters
        torch::nn::Hardtanh default_hardtanh;
        torch::Tensor output_default = default_hardtanh->forward(input);
        
        // Try with edge case parameters
        if (offset + 2*sizeof(float) <= Size) {
            float edge_min, edge_max;
            std::memcpy(&edge_min, Data + offset, sizeof(float));
            offset += sizeof(float);
            std::memcpy(&edge_max, Data + offset, sizeof(float));
            
            // Try with potentially problematic min/max values
            torch::nn::Hardtanh edge_hardtanh(torch::nn::HardtanhOptions().min_val(edge_min).max_val(edge_max));
            torch::Tensor output_edge = edge_hardtanh->forward(input);
            
            // Try with min > max (should swap them internally)
            torch::nn::Hardtanh swapped_hardtanh(torch::nn::HardtanhOptions().min_val(edge_max).max_val(edge_min));
            torch::Tensor output_swapped = swapped_hardtanh->forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}