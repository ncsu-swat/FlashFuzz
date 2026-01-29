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
        
        // Need at least some data to create a tensor
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor and convert to complex (irfft2 expects complex input)
        torch::Tensor real_part = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor has at least 2 dimensions for 2D FFT
        if (real_part.dim() < 2) {
            real_part = real_part.unsqueeze(0);
            if (real_part.dim() < 2) {
                real_part = real_part.unsqueeze(0);
            }
        }
        
        // Convert to float if needed and create complex tensor
        if (!real_part.is_floating_point()) {
            real_part = real_part.to(torch::kFloat32);
        }
        
        // Create complex tensor from real data (imaginary part from fuzzer or zeros)
        torch::Tensor imag_part;
        if (offset + 4 < Size) {
            imag_part = fuzzer_utils::createTensor(Data, Size, offset);
            // Match shape and type
            if (imag_part.numel() != real_part.numel()) {
                imag_part = torch::zeros_like(real_part);
            } else {
                imag_part = imag_part.reshape(real_part.sizes()).to(real_part.dtype());
            }
        } else {
            imag_part = torch::zeros_like(real_part);
        }
        
        torch::Tensor input = torch::complex(real_part, imag_part);
        
        // Parse norm parameter
        int64_t norm_value = 0;
        if (offset < Size) {
            norm_value = Data[offset++] % 4; // 0-3 for different norm options
        }
        
        // Create norm string based on the value
        std::optional<std::string_view> norm = std::nullopt;
        if (norm_value == 1) {
            norm = "forward";
        } else if (norm_value == 2) {
            norm = "backward";
        } else if (norm_value == 3) {
            norm = "ortho";
        }
        
        // Parse whether to use custom s (output size)
        bool use_custom_s = false;
        std::vector<int64_t> s_vec;
        if (offset + 2 < Size) {
            use_custom_s = Data[offset++] % 2 == 1;
            if (use_custom_s) {
                // s must have exactly 2 elements for irfft2
                int64_t s0 = 1 + (Data[offset++] % 64); // Reasonable output sizes
                int64_t s1 = 1 + (offset < Size ? Data[offset++] % 64 : 8);
                s_vec = {s0, s1};
            }
        }
        
        // Parse whether to use custom dims
        bool use_custom_dims = false;
        int64_t dim_h = -2;
        int64_t dim_w = -1;
        if (offset < Size) {
            use_custom_dims = Data[offset++] % 2 == 1;
            if (use_custom_dims && input.dim() >= 2) {
                // Choose valid dims within tensor dimensions
                int64_t ndim = input.dim();
                if (offset + 1 < Size) {
                    dim_h = -(1 + (Data[offset++] % std::min(ndim, (int64_t)4)));
                    dim_w = -(1 + (Data[offset++] % std::min(ndim, (int64_t)4)));
                    // Ensure dims are different
                    if (dim_h == dim_w) {
                        dim_w = (dim_h == -1) ? -2 : -1;
                    }
                }
            }
        }
        
        // Apply irfft2 operation
        torch::Tensor output;
        
        try {
            if (use_custom_s) {
                c10::IntArrayRef s_array(s_vec.data(), s_vec.size());
                output = torch::fft::irfft2(input, s_array, {dim_h, dim_w}, norm);
            } else {
                output = torch::fft::irfft2(input, std::nullopt, {dim_h, dim_w}, norm);
            }
        } catch (const c10::Error&) {
            // Expected errors from invalid parameters - try with defaults
            output = torch::fft::irfft2(input);
        }
        
        // Perform some operation on the output to ensure it's used
        auto sum = output.sum();
        
        // Test gradient computation
        if (offset < Size && Data[offset] % 4 == 0) {
            torch::Tensor grad_real = real_part.detach().clone().requires_grad_(true);
            torch::Tensor grad_imag = imag_part.detach().clone().requires_grad_(true);
            torch::Tensor grad_input = torch::complex(grad_real, grad_imag);
            
            try {
                auto out = torch::fft::irfft2(grad_input);
                auto sum_grad = out.sum();
                sum_grad.backward();
            } catch (const c10::Error&) {
                // Gradient computation may fail for some configurations
            }
        }
        
        // Test with hermitian input (valid for irfft2)
        if (offset + 1 < Size && Data[offset] % 3 == 0) {
            try {
                // Create a simple hermitian-like tensor
                auto small_real = torch::randn({2, 3});
                auto small_imag = torch::randn({2, 3});
                auto small_complex = torch::complex(small_real, small_imag);
                auto out = torch::fft::irfft2(small_complex);
                (void)out.sum().item<float>();
            } catch (const c10::Error&) {
                // May fail, that's ok
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