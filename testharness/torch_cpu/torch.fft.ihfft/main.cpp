#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

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
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // ihfft requires at least 1-dimensional real-valued input
        if (input.dim() == 0) {
            input = input.unsqueeze(0);
        }
        
        // ihfft expects a real-valued input tensor
        if (input.is_complex()) {
            input = torch::real(input);
        }
        
        // Ensure input is floating point
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Get a dimension to apply ihfft along (valid range: 0 to dim-1, or use -1)
        int64_t dim = -1;
        if (offset < Size) {
            uint8_t dim_selector = Data[offset++];
            if (input.dim() > 0) {
                dim = static_cast<int64_t>(dim_selector % input.dim());
            }
        }
        
        // Get norm parameter
        c10::string_view norm = "backward";
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++] % 3;
            if (norm_selector == 0) {
                norm = "backward";
            } else if (norm_selector == 1) {
                norm = "forward";
            } else {
                norm = "ortho";
            }
        }
        
        // Apply ihfft operation without n parameter
        try {
            torch::Tensor result = torch::fft::ihfft(input, c10::nullopt, dim, norm);
            (void)result;
        } catch (const std::exception &) {
            // Expected failures (e.g., invalid input) - continue fuzzing
        }
        
        // Try with n parameter (length of transformed axis)
        if (offset < Size) {
            // Get n value from fuzzer data
            int64_t n = static_cast<int64_t>(Data[offset++] % 64) + 1;
            
            try {
                torch::Tensor result = torch::fft::ihfft(input, c10::optional<int64_t>(n), dim, norm);
                (void)result;
            } catch (const std::exception &) {
                // Expected failures - continue fuzzing
            }
        }
        
        // Test with different dimensions if tensor has multiple dims
        if (input.dim() > 1 && offset < Size) {
            int64_t alt_dim = static_cast<int64_t>(Data[offset++] % input.dim());
            try {
                torch::Tensor result = torch::fft::ihfft(input, c10::nullopt, alt_dim, norm);
                (void)result;
            } catch (const std::exception &) {
                // Expected failures - continue
            }
        }
        
        // Test with contiguous vs non-contiguous input
        if (input.dim() >= 2) {
            try {
                torch::Tensor transposed = input.transpose(0, 1);
                torch::Tensor result = torch::fft::ihfft(transposed, c10::nullopt, -1, norm);
                (void)result;
            } catch (const std::exception &) {
                // Expected failures - continue
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