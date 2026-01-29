#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <ATen/Functions.h>

// --- Fuzzer Entry Point ---
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
        
        // Need at least a few bytes to create a tensor
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor and convert to complex for irfft
        torch::Tensor real_input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // irfft expects complex input - create a complex tensor
        // Convert real tensor to complex by adding zero imaginary part
        torch::Tensor input;
        try {
            // Ensure we have at least 1D tensor for FFT operations
            if (real_input.dim() == 0) {
                real_input = real_input.unsqueeze(0);
            }
            
            // Convert to float first if needed, then to complex
            torch::Tensor float_input = real_input.to(torch::kFloat32);
            input = torch::complex(float_input, torch::zeros_like(float_input));
        } catch (...) {
            return 0;  // Skip if conversion fails
        }
        
        // Extract parameters for irfft
        c10::optional<int64_t> n = c10::nullopt;
        int64_t dim = -1;
        c10::optional<c10::string_view> norm = c10::nullopt;
        
        // Parse n parameter if we have enough data
        if (offset + sizeof(int32_t) <= Size) {
            int32_t n_val;
            std::memcpy(&n_val, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            
            // Only set n if it's positive and reasonable
            if (n_val > 0 && n_val < 4096) {
                n = static_cast<int64_t>(n_val);
            }
        }
        
        // Parse dim parameter if we have enough data
        if (offset + sizeof(int8_t) <= Size) {
            int8_t dim_val = static_cast<int8_t>(Data[offset++]);
            
            // Ensure dim is within valid range for the tensor
            if (input.dim() > 0) {
                dim = dim_val % input.dim();
                if (dim < 0) {
                    dim += input.dim();
                }
            }
        }
        
        // Parse norm parameter if we have enough data
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++];
            switch (norm_selector % 4) {
                case 0:
                    norm = c10::nullopt;  // default
                    break;
                case 1:
                    norm = "backward";
                    break;
                case 2:
                    norm = "forward";
                    break;
                case 3:
                    norm = "ortho";
                    break;
            }
        }
        
        // Apply irfft operation
        torch::Tensor output;
        
        try {
            output = torch::fft::irfft(input, n, dim, norm);
        } catch (const c10::Error&) {
            // Shape/dimension errors are expected for some inputs
            return 0;
        }
        
        // Ensure the output is used to prevent optimization
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
    return 0; // keep the input
}