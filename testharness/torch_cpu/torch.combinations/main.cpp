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
        
        // Extract parameters for combinations
        // We need at least 2 bytes for r and with_replacement
        if (offset + 2 > Size) {
            return 0;
        }
        
        // Parse r parameter (number of elements to combine)
        int64_t r = 0;
        if (input.dim() > 0) {
            // r should be between 0 and the size of the first dimension
            uint8_t r_byte = Data[offset++];
            r = r_byte % (input.size(0) + 1);
        }
        
        // Parse with_replacement parameter
        bool with_replacement = (offset < Size) ? (Data[offset++] & 1) : false;
        
        // Call torch.combinations
        torch::Tensor result;
        try {
            result = torch::combinations(input, r, with_replacement);
        } catch (const c10::Error& e) {
            // PyTorch specific exceptions are expected and not a bug
            return 0;
        }
        
        // Try to access elements of the result tensor to ensure it's valid
        if (result.defined() && result.numel() > 0) {
            auto accessor = result.accessor<float, 2>();
            volatile float val = accessor[0][0];
            (void)val;
        }
        
        // Try different r values if we have more data
        if (offset + 1 < Size && input.dim() > 0) {
            uint8_t r2_byte = Data[offset++];
            int64_t r2 = r2_byte % (input.size(0) + 1);
            
            if (r2 != r) {
                try {
                    torch::Tensor result2 = torch::combinations(input, r2, with_replacement);
                    if (result2.defined() && result2.numel() > 0) {
                        auto accessor = result2.accessor<float, 2>();
                        volatile float val = accessor[0][0];
                        (void)val;
                    }
                } catch (const c10::Error& e) {
                    // Expected exceptions are fine
                }
            }
        }
        
        // Try with the opposite value of with_replacement
        try {
            torch::Tensor result3 = torch::combinations(input, r, !with_replacement);
            if (result3.defined() && result3.numel() > 0) {
                auto accessor = result3.accessor<float, 2>();
                volatile float val = accessor[0][0];
                (void)val;
            }
        } catch (const c10::Error& e) {
            // Expected exceptions are fine
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}