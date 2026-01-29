#include "fuzzer_utils.h" // General fuzzing utilities
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <tuple>

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
        
        // Create input tensor - hfft expects complex input
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input is complex (hfft operates on Hermitian symmetric input)
        // Convert to complex if not already
        if (!input.is_complex()) {
            // Create a complex tensor from real input
            // Using real part from input and zeros for imaginary
            input = torch::complex(input, torch::zeros_like(input));
        }
        
        // Ensure we have at least 1D tensor for FFT
        if (input.dim() == 0) {
            input = input.unsqueeze(0);
        }
        
        // Extract parameters for hfft if we have more data
        std::optional<int64_t> n_opt = std::nullopt;
        int64_t dim = -1; // Default dimension (last dim)
        std::optional<std::string> norm = std::nullopt;
        
        // Parse n parameter if we have enough data
        if (offset + sizeof(int64_t) <= Size) {
            int64_t raw_n;
            std::memcpy(&raw_n, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);

            // Bound n to a reasonable size to avoid huge allocations
            constexpr int64_t kMaxLength = 4096;
            int64_t abs_n = (raw_n == std::numeric_limits<int64_t>::min())
                                ? std::numeric_limits<int64_t>::max()
                                : std::abs(raw_n);
            if (abs_n > 0) {
                int64_t bounded_n = 1 + (abs_n % kMaxLength);
                n_opt = bounded_n;
            }
        }
        
        // Parse dim parameter if we have enough data
        if (offset + sizeof(int8_t) <= Size) {
            int8_t raw_dim;
            std::memcpy(&raw_dim, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            
            // Ensure dim is valid for the input tensor
            if (input.dim() > 0) {
                dim = raw_dim % input.dim();
                if (dim < 0) dim += input.dim();
            }
        }
        
        // Parse norm parameter if we have enough data
        if (offset < Size) {
            uint8_t norm_selector = Data[offset] % 4;
            offset++;
            switch (norm_selector) {
                case 0:
                    norm = std::nullopt;  // Default
                    break;
                case 1:
                    norm = "forward";
                    break;
                case 2:
                    norm = "backward";
                    break;
                case 3:
                    norm = "ortho";
                    break;
            }
        }
        
        // Apply hfft operation
        // hfft computes the FFT of a Hermitian symmetric input signal
        // The output is real-valued
        torch::Tensor output;
        
        // Inner try-catch for expected failures (shape mismatches, etc.)
        try {
            output = torch::fft::hfft(input, n_opt, dim, norm);
        } catch (const c10::Error&) {
            // Expected error (e.g., invalid shapes), silently ignore
            return 0;
        }
        
        // Force evaluation of the output tensor
        if (output.defined() && output.numel() > 0) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // Keep the input
}