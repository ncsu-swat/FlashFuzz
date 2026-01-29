#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Need at least some data to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor - hfft2 works on complex input
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 2 dimensions for 2D FFT
        if (input.dim() < 2) {
            input = input.unsqueeze(0).unsqueeze(0);
        } else if (input.dim() < 2) {
            input = input.unsqueeze(0);
        }
        
        // Convert to complex type if not already
        if (!input.is_complex()) {
            input = torch::complex(input, torch::zeros_like(input));
        }
        
        // Parse optional parameters if we have more data
        int64_t s_h = -1;
        int64_t s_w = -1;
        int64_t dim_h = -2;
        int64_t dim_w = -1;
        std::string norm_str = "backward";
        
        // Parse s (output shape) if we have enough data
        if (offset + 16 <= Size) {
            std::memcpy(&s_h, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            std::memcpy(&s_w, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Clamp to reasonable values to avoid memory issues
            s_h = std::abs(s_h) % 256 + 1;
            s_w = std::abs(s_w) % 256 + 1;
        }
        
        // Parse dim (dimensions to transform) if we have enough data
        if (offset + 16 <= Size) {
            std::memcpy(&dim_h, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            std::memcpy(&dim_w, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Parse normalization if we have enough data
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++];
            switch (norm_selector % 3) {
                case 0: norm_str = "backward"; break;
                case 1: norm_str = "forward"; break;
                case 2: norm_str = "ortho"; break;
            }
        }
        
        // Prepare parameters - keep vectors alive for the duration of use
        std::vector<int64_t> s_vec;
        std::vector<int64_t> dim_vec;
        
        // Determine which variant to call based on remaining data
        uint8_t variant = (offset < Size) ? Data[offset] % 4 : 0;
        
        torch::Tensor output;
        
        try {
            // Normalize dim values to valid range
            int64_t ndim = input.dim();
            dim_h = ((dim_h % ndim) + ndim) % ndim;
            dim_w = ((dim_w % ndim) + ndim) % ndim;
            if (dim_h == dim_w) {
                dim_w = (dim_h + 1) % ndim;
            }
            
            switch (variant) {
                case 0: {
                    // Call with all parameters
                    s_vec = {s_h, s_w};
                    dim_vec = {dim_h, dim_w};
                    output = torch::fft::hfft2(input, s_vec, dim_vec, norm_str);
                    break;
                }
                case 1: {
                    // Call without s
                    dim_vec = {dim_h, dim_w};
                    output = torch::fft::hfft2(input, c10::nullopt, dim_vec, norm_str);
                    break;
                }
                case 2: {
                    // Call with s, default dim
                    s_vec = {s_h, s_w};
                    output = torch::fft::hfft2(input, s_vec, {-2, -1}, norm_str);
                    break;
                }
                default: {
                    // Call with minimal parameters
                    output = torch::fft::hfft2(input);
                    break;
                }
            }
        } catch (const c10::Error&) {
            // Expected failures (shape mismatches, invalid dims, etc.) - silently ignore
            return 0;
        } catch (const std::runtime_error&) {
            // Expected runtime errors - silently ignore
            return 0;
        }
        
        // Access output tensor to ensure computation is performed
        if (output.defined() && output.numel() > 0) {
            auto sum = output.sum().item<double>();
            (void)sum; // Prevent unused variable warning
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}