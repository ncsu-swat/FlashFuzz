#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for atanh
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply atanh operation
        // Note: atanh is defined for values in (-1, 1), but we let PyTorch handle edge cases
        torch::Tensor result = torch::atanh(input);
        
        // Force computation to ensure the operation is actually executed
        result.numel();
        
        // Try in-place version if there's more data
        if (offset < Size) {
            torch::Tensor input_copy = input.clone();
            input_copy.atanh_();
            input_copy.numel();
        }
        
        // Try with different options if we have more data
        if (offset + 1 < Size) {
            uint8_t variant_selector = Data[offset++];
            
            // Try out parameter version
            if (variant_selector & 0x1) {
                torch::Tensor out = torch::empty_like(input);
                torch::atanh_out(out, input);
                out.numel();
            }
            
            // Try with different input types/shapes
            if (variant_selector & 0x2 && offset < Size) {
                try {
                    torch::Tensor another_input = fuzzer_utils::createTensor(Data, Size, offset);
                    torch::Tensor another_result = torch::atanh(another_input);
                    another_result.numel();
                } catch (const std::exception &) {
                    // Silently ignore - inner exception for expected failures
                }
            }
            
            // Try with explicitly clamped input to test valid domain
            if (variant_selector & 0x4) {
                try {
                    // Clamp input to valid domain (-1, 1) to test the main computation path
                    torch::Tensor clamped = torch::clamp(input, -0.999, 0.999);
                    torch::Tensor clamped_result = torch::atanh(clamped);
                    clamped_result.numel();
                } catch (const std::exception &) {
                    // Silently ignore
                }
            }
            
            // Try with different dtypes
            if (variant_selector & 0x8) {
                try {
                    torch::Tensor float_input = input.to(torch::kFloat32);
                    torch::Tensor float_result = torch::atanh(float_input);
                    float_result.numel();
                } catch (const std::exception &) {
                    // Silently ignore - dtype conversion may fail
                }
            }
            
            if (variant_selector & 0x10) {
                try {
                    torch::Tensor double_input = input.to(torch::kFloat64);
                    torch::Tensor double_result = torch::atanh(double_input);
                    double_result.numel();
                } catch (const std::exception &) {
                    // Silently ignore
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}