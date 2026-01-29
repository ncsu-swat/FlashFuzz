#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least some data to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor - ifft2 works on 2D+ tensors
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor has at least 2 dimensions for ifft2
        if (input_tensor.dim() < 2) {
            // Reshape to 2D if needed
            auto numel = input_tensor.numel();
            if (numel < 2) {
                input_tensor = torch::zeros({2, 2});
            } else {
                int64_t side = static_cast<int64_t>(std::sqrt(static_cast<double>(numel)));
                if (side < 2) side = 2;
                input_tensor = input_tensor.flatten().slice(0, 0, side * side).reshape({side, side});
            }
        }
        
        // Convert to complex type if not already (FFT requires complex or will convert)
        if (!input_tensor.is_complex()) {
            input_tensor = torch::complex(input_tensor, torch::zeros_like(input_tensor));
        }
        
        // Parse optional parameters from the remaining data
        c10::optional<at::IntArrayRef> s = c10::nullopt;
        std::vector<int64_t> s_vec;
        at::IntArrayRef dim = {-2, -1};  // Default dimensions for ifft2
        std::vector<int64_t> dim_vec;
        c10::optional<c10::string_view> norm = c10::nullopt;
        
        if (offset < Size) {
            uint8_t param_flags = Data[offset++];
            
            // Parse output sizes 's' parameter
            if ((param_flags & 0x1) && offset + 1 < Size) {
                int64_t s0 = 1 + (Data[offset++] % 16);  // Size between 1 and 16
                int64_t s1 = 1 + (Data[offset++] % 16);
                s_vec = {s0, s1};
                s = at::IntArrayRef(s_vec);
            }
            
            // Parse custom dimensions
            if ((param_flags & 0x2) && offset + 1 < Size && input_tensor.dim() >= 2) {
                int64_t ndim = input_tensor.dim();
                int64_t d0 = static_cast<int64_t>(Data[offset++] % ndim) - ndim;
                int64_t d1 = static_cast<int64_t>(Data[offset++] % ndim) - ndim;
                if (d0 != d1) {
                    dim_vec = {d0, d1};
                    dim = at::IntArrayRef(dim_vec);
                }
            }
            
            // Parse normalization mode
            if ((param_flags & 0x4) && offset < Size) {
                uint8_t norm_choice = Data[offset++] % 4;
                switch (norm_choice) {
                    case 0: norm = c10::nullopt; break;  // default "backward"
                    case 1: norm = "backward"; break;
                    case 2: norm = "ortho"; break;
                    case 3: norm = "forward"; break;
                }
            }
        }
        
        // Apply ifft2 operation with various parameter combinations
        torch::Tensor output;
        
        try {
            output = torch::fft::ifft2(input_tensor, s, dim, norm);
        } catch (const std::exception &) {
            // Shape mismatches or invalid dim combinations are expected
            return 0;
        }
        
        // Verify output is valid by performing operations
        auto sum = output.sum();
        
        // Access real and imaginary parts
        auto real_part = torch::real(output);
        auto imag_part = torch::imag(output);
        
        // Compute magnitude
        auto magnitude = torch::abs(output);
        
        // Try round-trip: fft2(ifft2(x)) should approximate x
        try {
            auto roundtrip = torch::fft::fft2(output, s, dim, norm);
            (void)roundtrip;
        } catch (const std::exception &) {
            // Ignore errors in round-trip test
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}