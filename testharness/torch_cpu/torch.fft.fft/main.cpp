#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        // FFT requires at least 1D input
        if (input_tensor.dim() == 0) {
            input_tensor = input_tensor.unsqueeze(0);
        }
        
        // Parse parameters from remaining fuzzer data
        int64_t n = -1;  // Default: use input size
        int64_t dim = -1; // Default: last dimension
        c10::optional<std::string> norm_opt = c10::nullopt;
        
        // Parse control byte for parameter combinations
        uint8_t control = 0;
        if (offset < Size) {
            control = Data[offset++];
        }
        
        // Parse n parameter - limit to reasonable range to avoid OOM
        if (offset + sizeof(int16_t) <= Size) {
            int16_t n_raw;
            std::memcpy(&n_raw, Data + offset, sizeof(int16_t));
            offset += sizeof(int16_t);
            // Limit n to reasonable range [1, 1024]
            if (n_raw > 0) {
                n = (std::abs(n_raw) % 1024) + 1;
            }
        }
        
        // Parse dim parameter
        if (offset < Size && input_tensor.dim() > 0) {
            int8_t dim_raw = static_cast<int8_t>(Data[offset++]);
            // Map to valid dimension range
            dim = dim_raw % input_tensor.dim();
            if (dim < 0) {
                dim += input_tensor.dim();
            }
        }
        
        // Parse norm parameter
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++] % 4;
            if (norm_selector == 0) {
                norm_opt = "forward";
            } else if (norm_selector == 1) {
                norm_opt = "backward";
            } else if (norm_selector == 2) {
                norm_opt = "ortho";
            }
            // selector == 3: use default (nullopt)
        }
        
        torch::Tensor result;
        
        // Select which parameter combination to test based on control byte
        uint8_t test_case = control % 8;
        
        try {
            switch (test_case) {
                case 0:
                    // Basic FFT with default parameters
                    result = torch::fft::fft(input_tensor);
                    break;
                case 1:
                    // FFT with specified n
                    if (n > 0) {
                        result = torch::fft::fft(input_tensor, n);
                    } else {
                        result = torch::fft::fft(input_tensor);
                    }
                    break;
                case 2:
                    // FFT with specified n and dim
                    if (n > 0 && dim >= 0) {
                        result = torch::fft::fft(input_tensor, n, dim);
                    } else {
                        result = torch::fft::fft(input_tensor);
                    }
                    break;
                case 3:
                    // FFT with all parameters
                    if (n > 0 && dim >= 0 && norm_opt.has_value()) {
                        result = torch::fft::fft(input_tensor, n, dim, norm_opt.value());
                    } else {
                        result = torch::fft::fft(input_tensor);
                    }
                    break;
                case 4:
                    // FFT with just dim
                    if (dim >= 0) {
                        result = torch::fft::fft(input_tensor, c10::nullopt, dim);
                    } else {
                        result = torch::fft::fft(input_tensor);
                    }
                    break;
                case 5:
                    // FFT with just norm
                    if (norm_opt.has_value()) {
                        result = torch::fft::fft(input_tensor, c10::nullopt, -1, norm_opt.value());
                    } else {
                        result = torch::fft::fft(input_tensor);
                    }
                    break;
                case 6:
                    // FFT with dim and norm
                    if (dim >= 0 && norm_opt.has_value()) {
                        result = torch::fft::fft(input_tensor, c10::nullopt, dim, norm_opt.value());
                    } else {
                        result = torch::fft::fft(input_tensor);
                    }
                    break;
                case 7:
                    // FFT with n and norm
                    if (n > 0 && norm_opt.has_value()) {
                        result = torch::fft::fft(input_tensor, n, -1, norm_opt.value());
                    } else {
                        result = torch::fft::fft(input_tensor);
                    }
                    break;
            }
        } catch (const c10::Error&) {
            // Expected errors for invalid shapes/params - silently continue
        } catch (const std::runtime_error&) {
            // Expected errors - silently continue
        }
        
        // Force computation if we got a result
        if (result.defined() && result.numel() > 0) {
            volatile auto sum = result.abs().sum().item<float>();
            (void)sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}