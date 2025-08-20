#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse rcond parameter if we have more data
        double rcond = 1e-15; // Default value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&rcond, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure rcond is a reasonable value
            if (std::isnan(rcond) || std::isinf(rcond)) {
                rcond = 1e-15;
            }
        }
        
        // Parse hermitian parameter if we have more data
        bool hermitian = false;
        if (offset < Size) {
            hermitian = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Call torch::pinv with different parameter combinations
        torch::Tensor result;
        
        // Try different combinations of parameters
        if (offset < Size) {
            uint8_t param_selector = Data[offset++] % 4;
            
            switch (param_selector) {
                case 0:
                    // Just the input tensor
                    result = torch::pinv(input);
                    break;
                case 1:
                    // Input tensor and rcond
                    result = torch::pinv(input, rcond);
                    break;
                case 2:
                    // Input tensor and hermitian flag
                    result = torch::pinv(input, rcond, hermitian);
                    break;
                case 3:
                    // All parameters
                    result = torch::pinv(input, rcond, hermitian);
                    break;
            }
        } else {
            // Default case if we don't have enough data
            result = torch::pinv(input);
        }
        
        // Basic validation - just check that the result is a valid tensor
        if (result.defined() && !result.sizes().empty()) {
            // Access some elements to ensure computation completed
            if (result.numel() > 0) {
                auto first_elem = result.flatten()[0];
                // Use the value to prevent compiler optimization
                if (std::isnan(first_elem.item<double>())) {
                    // This is just to use the value, not actually checking for NaN
                    return 0;
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