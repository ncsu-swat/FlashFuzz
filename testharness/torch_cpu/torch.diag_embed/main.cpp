#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for the input tensor and parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for diag_embed if there's data left
        int64_t offset_param = 0;
        int64_t dim1_param = -2;
        int64_t dim2_param = -1;
        
        if (offset < Size) {
            // Use modulo to keep offset parameter reasonable (-10 to 10)
            offset_param = static_cast<int64_t>(Data[offset] % 21) - 10;
            offset++;
        }
        
        if (offset < Size) {
            // dim1 and dim2 need to be valid for output tensor dimensions
            // Output has input.dim() + 1 dimensions
            // Keep dim1 in range [-output_dim, output_dim-1]
            int64_t output_dim = input.dim() + 1;
            if (output_dim > 0) {
                dim1_param = static_cast<int64_t>(Data[offset] % (2 * output_dim)) - output_dim;
            }
            offset++;
        }
        
        if (offset < Size) {
            int64_t output_dim = input.dim() + 1;
            if (output_dim > 0) {
                dim2_param = static_cast<int64_t>(Data[offset] % (2 * output_dim)) - output_dim;
            }
            offset++;
        }
        
        // Ensure dim1 != dim2 (required by diag_embed)
        if (dim1_param == dim2_param) {
            dim2_param = (dim1_param == -1) ? -2 : -1;
        }
        
        // Determine which variant to try based on remaining data
        int variant = (offset < Size) ? (Data[offset] % 4) : 0;
        
        torch::Tensor result;
        
        try {
            switch (variant) {
                case 0:
                    // Use all three parameters
                    result = torch::diag_embed(input, offset_param, dim1_param, dim2_param);
                    break;
                case 1:
                    // Use two parameters (offset and dim1)
                    result = torch::diag_embed(input, offset_param, dim1_param);
                    break;
                case 2:
                    // Use just offset parameter
                    result = torch::diag_embed(input, offset_param);
                    break;
                default:
                    // Use default parameters
                    result = torch::diag_embed(input);
                    break;
            }
            
            // Force evaluation of the result
            result.sum().item<float>();
        }
        catch (...) {
            // Silently catch expected failures (shape mismatches, invalid dims)
        }
        
        // Try with default parameters on a fresh attempt
        try {
            torch::Tensor result2 = torch::diag_embed(input);
            result2.sum().item<float>();
        }
        catch (...) {
            // Silently catch expected failures
        }
        
        // Try with a 1D tensor specifically (common use case)
        try {
            if (input.numel() > 0) {
                torch::Tensor flat_input = input.flatten();
                torch::Tensor result3 = torch::diag_embed(flat_input);
                result3.sum().item<float>();
                
                // Also try with offset
                torch::Tensor result4 = torch::diag_embed(flat_input, offset_param);
                result4.sum().item<float>();
            }
        }
        catch (...) {
            // Silently catch expected failures
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}