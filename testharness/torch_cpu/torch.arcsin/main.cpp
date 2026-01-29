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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply arcsin operation
        // Note: arcsin is defined for values in [-1, 1], values outside produce NaN
        torch::Tensor result = torch::arcsin(input);
        
        // Force computation to ensure the operation is actually executed
        (void)result.sum().item<float>();
        
        // Try in-place version if there's enough data to decide
        if (offset < Size) {
            bool use_inplace = Data[offset++] % 2 == 0;
            if (use_inplace) {
                try {
                    torch::Tensor input_copy = input.clone();
                    input_copy.arcsin_();
                    (void)input_copy.sum().item<float>();
                } catch (...) {
                    // In-place may fail on certain tensor types, ignore
                }
            }
        }
        
        // Try with out parameter if there's more data
        if (offset + 1 < Size) {
            bool use_out = Data[offset++] % 2 == 0;
            if (use_out) {
                try {
                    // Create an output tensor with same shape and dtype
                    torch::Tensor out = torch::empty_like(input);
                    torch::arcsin_out(out, input);
                    (void)out.sum().item<float>();
                } catch (...) {
                    // arcsin_out may fail with certain dtype combinations, ignore
                }
            }
        }
        
        // Test with different dtypes
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++] % 4;
            try {
                torch::Tensor converted;
                switch (dtype_selector) {
                    case 0:
                        converted = input.to(torch::kFloat32);
                        break;
                    case 1:
                        converted = input.to(torch::kFloat64);
                        break;
                    case 2:
                        converted = input.to(torch::kFloat16);
                        break;
                    default:
                        converted = input;
                        break;
                }
                torch::Tensor dtype_result = torch::arcsin(converted);
                (void)dtype_result.sum().item<float>();
            } catch (...) {
                // Type conversion or arcsin on converted may fail, ignore
            }
        }
        
        // Test with clamped input to ensure valid domain [-1, 1]
        if (offset < Size && Data[offset++] % 2 == 0) {
            try {
                torch::Tensor clamped = torch::clamp(input.to(torch::kFloat32), -1.0, 1.0);
                torch::Tensor valid_result = torch::arcsin(clamped);
                (void)valid_result.sum().item<float>();
            } catch (...) {
                // Ignore failures
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