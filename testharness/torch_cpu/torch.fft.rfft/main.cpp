#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr, cout
#include <cstdint>        // For uint64_t

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
        
        // Need at least a few bytes to create a tensor
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // rfft requires real-valued floating point input
        // Convert to float if it's not already a floating point type
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // If input is complex, take real part (rfft expects real input)
        if (input.is_complex()) {
            input = torch::real(input);
        }
        
        // Ensure tensor has at least 1 dimension for FFT
        if (input.dim() == 0) {
            input = input.unsqueeze(0);
        }
        
        // Parse dim parameter first (before n, since n depends on dimension size)
        int64_t dim = -1;  // Default to last dimension
        if (offset + sizeof(int8_t) <= Size) {
            int8_t dim_byte = static_cast<int8_t>(Data[offset++]);
            // Make dim valid by taking modulo of number of dimensions
            dim = ((dim_byte % input.dim()) + input.dim()) % input.dim();
        }
        
        // Parse n parameter (optional, controls output size)
        c10::optional<int64_t> n = c10::nullopt;
        if (offset + sizeof(uint8_t) <= Size) {
            uint8_t n_byte = Data[offset++];
            if (n_byte > 0) {
                // Limit n to reasonable range based on input dimension size
                int64_t dim_size = input.size(dim);
                // n can be 1 to 2*dim_size (allow some padding)
                int64_t max_n = std::max<int64_t>(1, std::min<int64_t>(dim_size * 2, 1024));
                n = 1 + (n_byte % max_n);
            }
            // If n_byte == 0, keep n as nullopt (use default)
        }
        
        // Parse norm parameter
        c10::optional<std::string> norm = c10::nullopt;
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++];
            switch (norm_selector % 4) {
                case 0: norm = c10::nullopt; break;  // Default (backward)
                case 1: norm = "forward"; break;
                case 2: norm = "ortho"; break;
                case 3: norm = "backward"; break;
            }
        }
        
        // Apply rfft operation
        torch::Tensor result;
        try {
            result = torch::fft::rfft(input, n, dim, norm);
        } catch (const c10::Error&) {
            // Shape/dimension errors are expected for some inputs
            return 0;
        }
        
        // Verify result is complex
        if (!result.is_complex()) {
            std::cerr << "Unexpected: rfft result is not complex" << std::endl;
        }
        
        // Perform operation on result to ensure it's computed
        auto abs_result = torch::abs(result);
        auto sum = abs_result.sum();
        (void)sum;
        
        // Try inverse operation to test round-trip
        try {
            if (result.numel() > 0) {
                // For irfft, we need to specify the original size if n was used
                c10::optional<int64_t> irfft_n = c10::nullopt;
                if (n.has_value()) {
                    irfft_n = n;
                }
                auto inverse = torch::fft::irfft(result, irfft_n, dim, norm);
                (void)inverse;
            }
        } catch (const c10::Error&) {
            // Inverse may fail for some parameter combinations, that's ok
        }
        
        // Test rfft with different input shapes/types for better coverage
        if (offset < Size && (Data[offset] % 4) == 0) {
            // Test with double precision
            try {
                auto double_input = input.to(torch::kFloat64);
                auto double_result = torch::fft::rfft(double_input, n, dim, norm);
                (void)double_result;
            } catch (const c10::Error&) {
                // Expected for some inputs
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}