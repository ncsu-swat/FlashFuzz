#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for torch.erfc
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.erfc operation
        torch::Tensor result = torch::erfc(input);
        
        // Try some variants of the operation
        if (offset + 1 < Size) {
            // Try in-place version if available
            torch::Tensor input_copy = input.clone();
            input_copy.erfc_();
            
            // Try with different output types if possible
            if (input.scalar_type() != torch::kBool) {
                torch::Tensor result_float = torch::erfc(input.to(torch::kFloat));
                torch::Tensor result_double = torch::erfc(input.to(torch::kDouble));
                
                // Try with complex inputs if we have enough data
                if (offset + 2 < Size) {
                    try {
                        torch::Tensor complex_input = input.to(torch::kComplexFloat);
                        torch::Tensor complex_result = torch::erfc(complex_input);
                    } catch (const std::exception&) {
                        // Ignore exceptions from complex conversion
                    }
                }
            }
        }
        
        // Try with empty tensor
        try {
            torch::Tensor empty_tensor = torch::empty({0});
            torch::Tensor empty_result = torch::erfc(empty_tensor);
        } catch (const std::exception&) {
            // Ignore exceptions from empty tensor
        }
        
        // Try with scalar tensor
        if (offset + 1 < Size) {
            try {
                torch::Tensor scalar_tensor = torch::tensor(static_cast<float>(Data[offset]));
                torch::Tensor scalar_result = torch::erfc(scalar_tensor);
            } catch (const std::exception&) {
                // Ignore exceptions from scalar tensor
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