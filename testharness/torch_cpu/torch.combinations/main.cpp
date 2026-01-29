#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor and parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // torch::combinations requires a 1D tensor - flatten if needed
        input = input.flatten();
        
        // Limit input size to avoid combinatorial explosion
        // combinations(n, r) can produce n!/(r!(n-r)!) outputs
        if (input.numel() > 20) {
            input = input.slice(0, 0, 20);
        }
        
        // Need at least 2 bytes for r and with_replacement
        if (offset + 2 > Size) {
            return 0;
        }
        
        // Parse r parameter (number of elements to combine)
        int64_t r = 0;
        if (input.numel() > 0) {
            // r should be between 0 and the size of the tensor
            uint8_t r_byte = Data[offset++];
            r = r_byte % (input.numel() + 1);
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
        
        // Verify result is valid by computing sum (works for any dtype)
        if (result.defined() && result.numel() > 0) {
            try {
                auto sum = result.sum();
                volatile float val = sum.item<float>();
                (void)val;
            } catch (...) {
                // Ignore accessor errors for non-float types
            }
        }
        
        // Try different r values if we have more data
        if (offset + 1 < Size && input.numel() > 0) {
            uint8_t r2_byte = Data[offset++];
            int64_t r2 = r2_byte % (input.numel() + 1);
            
            if (r2 != r) {
                try {
                    torch::Tensor result2 = torch::combinations(input, r2, with_replacement);
                    if (result2.defined() && result2.numel() > 0) {
                        auto sum = result2.sum();
                        volatile float val = sum.item<float>();
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
                auto sum = result3.sum();
                volatile float val = sum.item<float>();
                (void)val;
            }
        } catch (const c10::Error& e) {
            // Expected exceptions are fine
        }
        
        // Test edge case: r = 0 (should return empty combinations)
        try {
            torch::Tensor result_r0 = torch::combinations(input, 0, with_replacement);
            (void)result_r0;
        } catch (const c10::Error& e) {
            // Expected exceptions are fine
        }
        
        // Test edge case: r = input.numel() (all elements)
        if (input.numel() > 0 && input.numel() <= 10) {
            try {
                torch::Tensor result_all = torch::combinations(input, input.numel(), with_replacement);
                (void)result_all;
            } catch (const c10::Error& e) {
                // Expected exceptions are fine
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}