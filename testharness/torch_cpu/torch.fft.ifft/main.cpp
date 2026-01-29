#include "fuzzer_utils.h"
#include <iostream>
#include <optional>
#include <cstring>
#include <cstdlib>

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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // FFT requires floating point or complex input - convert to complex float
        if (!input_tensor.is_complex()) {
            input_tensor = input_tensor.to(torch::kComplexFloat);
        }
        
        // Parse FFT dimension parameter if we have more data
        int64_t dim = -1;
        if (offset + sizeof(int64_t) <= Size) {
            int64_t dim_val;
            std::memcpy(&dim_val, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            auto rank = input_tensor.dim();
            if (rank > 0) {
                // Clamp dim into valid range [-rank, rank-1]
                dim = dim_val % rank;
            }
        }
        
        // Parse norm parameter if we have more data
        std::optional<c10::string_view> norm = std::nullopt;
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++];
            switch (norm_selector % 4) {
                case 0: norm = "backward"; break;
                case 1: norm = "ortho"; break;
                case 2: norm = "forward"; break;
                case 3: norm = std::nullopt; break; // Also test default
            }
        }

        // Test basic ifft without n parameter
        try {
            torch::Tensor result = torch::fft::ifft(input_tensor, std::nullopt, dim, norm);
            // Use the result to prevent optimization
            auto sum = result.sum();
            (void)sum;
        } catch (const std::exception &) {
            // Expected for invalid shapes/dims
        }
        
        // Try with n parameter if we have more data
        if (offset + sizeof(int64_t) <= Size) {
            int64_t n_raw;
            std::memcpy(&n_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);

            // Keep n within a small positive range to avoid huge allocations
            int64_t n = 1 + (std::abs(n_raw) % 64);
            
            try {
                torch::Tensor result_with_n = torch::fft::ifft(input_tensor, n, dim, norm);
                auto sum_with_n = result_with_n.sum();
                (void)sum_with_n;
            } catch (const std::exception &) {
                // Expected for invalid parameters
            }
        }
        
        // Test with real-valued input (converted to float, not complex)
        try {
            torch::Tensor real_input = fuzzer_utils::createTensor(Data, Size, offset);
            real_input = real_input.to(torch::kFloat);
            torch::Tensor result_real = torch::fft::ifft(real_input, std::nullopt, dim, norm);
            auto sum_real = result_real.sum();
            (void)sum_real;
        } catch (const std::exception &) {
            // Expected for invalid inputs
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}