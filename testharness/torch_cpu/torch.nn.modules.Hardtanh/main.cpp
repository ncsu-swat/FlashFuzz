#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor and parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract min_val and max_val from the remaining data
        float min_val = -1.0f;
        float max_val = 1.0f;
        
        // If we have enough data left, use it to set min_val and max_val
        if (offset + sizeof(float) <= Size) {
            float extracted_val;
            std::memcpy(&extracted_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            min_val = extracted_val;
        }
        
        if (offset + sizeof(float) <= Size) {
            float extracted_val;
            std::memcpy(&extracted_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            max_val = extracted_val;
        }
        
        // Create Hardtanh module with various configurations
        torch::nn::Hardtanh hardtanh_module(torch::nn::HardtanhOptions().min_val(min_val).max_val(max_val));
        
        // Apply Hardtanh to the input tensor
        torch::Tensor output = hardtanh_module->forward(input);
        
        // Try functional version as well
        torch::Tensor output_functional = torch::nn::functional::hardtanh(
            input, 
            torch::nn::functional::HardtanhFuncOptions().min_val(min_val).max_val(max_val)
        );
        
        // Try inplace version if we have enough data left
        if (offset < Size) {
            uint8_t inplace_flag = Data[offset++];
            if (inplace_flag % 2 == 0) {
                torch::Tensor input_clone = input.clone();
                input_clone = torch::nn::functional::hardtanh(
                    input_clone, 
                    torch::nn::functional::HardtanhFuncOptions().min_val(min_val).max_val(max_val)
                );
            }
        }
        
        // Try with default parameters
        torch::nn::Hardtanh default_hardtanh;
        torch::Tensor output_default = default_hardtanh->forward(input);
        
        // Try with edge case parameters
        if (offset < Size) {
            uint8_t edge_case = Data[offset++];
            if (edge_case % 5 == 0) {
                // Case where min_val > max_val
                torch::nn::Hardtanh inverted_hardtanh(torch::nn::HardtanhOptions().min_val(max_val).max_val(min_val));
                torch::Tensor output_inverted = inverted_hardtanh->forward(input);
            } else if (edge_case % 5 == 1) {
                // Case with very large values
                torch::nn::Hardtanh large_hardtanh(torch::nn::HardtanhOptions().min_val(-1e10).max_val(1e10));
                torch::Tensor output_large = large_hardtanh->forward(input);
            } else if (edge_case % 5 == 2) {
                // Case with very small range
                torch::nn::Hardtanh small_hardtanh(torch::nn::HardtanhOptions().min_val(-1e-10).max_val(1e-10));
                torch::Tensor output_small = small_hardtanh->forward(input);
            } else if (edge_case % 5 == 3) {
                // Case with equal min and max
                torch::nn::Hardtanh equal_hardtanh(torch::nn::HardtanhOptions().min_val(min_val).max_val(min_val));
                torch::Tensor output_equal = equal_hardtanh->forward(input);
            } else {
                // Case with NaN values if supported by the tensor type
                if (input.scalar_type() == torch::kFloat || 
                    input.scalar_type() == torch::kDouble || 
                    input.scalar_type() == torch::kHalf) {
                    torch::Tensor nan_input = input.clone();
                    nan_input.index_put_({0}, std::numeric_limits<float>::quiet_NaN());
                    torch::Tensor output_nan = hardtanh_module->forward(nan_input);
                }
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