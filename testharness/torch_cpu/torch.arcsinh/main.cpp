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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply arcsinh operation
        torch::Tensor result = torch::arcsinh(input);
        
        // Try different overloads
        if (offset + 1 < Size) {
            // In-place version
            torch::Tensor input_copy = input.clone();
            input_copy.arcsinh_();
            
            // Out version
            torch::Tensor out = torch::empty_like(input);
            torch::arcsinh_out(out, input);
        }
        
        // Try with different options if we have more data
        if (offset + 2 < Size) {
            uint8_t option_byte = Data[offset++];
            
            // Try with different dtypes
            if (option_byte % 4 == 0) {
                torch::Tensor result_float = torch::arcsinh(input.to(torch::kFloat));
            } else if (option_byte % 4 == 1) {
                torch::Tensor result_double = torch::arcsinh(input.to(torch::kDouble));
            } else if (option_byte % 4 == 2) {
                // Try with complex numbers if possible
                try {
                    torch::Tensor result_complex = torch::arcsinh(input.to(torch::kComplexFloat));
                } catch (...) {
                    // Ignore conversion errors
                }
            } else {
                // Try with half precision if available
                try {
                    torch::Tensor result_half = torch::arcsinh(input.to(torch::kHalf));
                } catch (...) {
                    // Ignore conversion errors
                }
            }
        }
        
        // Try with extreme values if we have more data
        if (offset + 1 < Size) {
            uint8_t extreme_byte = Data[offset++];
            
            try {
                // Create tensors with extreme values
                std::vector<int64_t> shape = {1, 1};
                torch::Tensor extreme;
                
                if (extreme_byte % 5 == 0) {
                    // Very large positive value
                    extreme = torch::full(shape, 1e38, torch::kFloat);
                } else if (extreme_byte % 5 == 1) {
                    // Very large negative value
                    extreme = torch::full(shape, -1e38, torch::kFloat);
                } else if (extreme_byte % 5 == 2) {
                    // Infinity
                    extreme = torch::full(shape, std::numeric_limits<float>::infinity(), torch::kFloat);
                } else if (extreme_byte % 5 == 3) {
                    // NaN
                    extreme = torch::full(shape, std::numeric_limits<float>::quiet_NaN(), torch::kFloat);
                } else {
                    // Zero
                    extreme = torch::zeros(shape, torch::kFloat);
                }
                
                torch::Tensor extreme_result = torch::arcsinh(extreme);
            } catch (...) {
                // Ignore errors with extreme values
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
