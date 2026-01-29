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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse FFT parameters from the remaining data
        int64_t n = 0;
        int64_t dim = -1;
        std::string norm = "backward";
        
        // Parse n (size of FFT)
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&n, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Clamp n to reasonable range to avoid OOM
            n = std::abs(n) % 1024 + 1;
        }
        
        // Parse dim (dimension along which to perform FFT)
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Normalize dim to valid range
            if (input_tensor.dim() > 0) {
                dim = dim % input_tensor.dim();
            } else {
                dim = -1;
            }
        }
        
        // Parse norm type
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++];
            switch (norm_selector % 3) {
                case 0:
                    norm = "backward";
                    break;
                case 1:
                    norm = "forward";
                    break;
                case 2:
                    norm = "ortho";
                    break;
            }
        }
        
        // Apply FFT operations with different parameters
        try {
            // FFT with default parameters
            auto result1 = torch::fft::fft(input_tensor);
            
            // FFT with specified n
            auto result2 = torch::fft::fft(input_tensor, n);
            
            // FFT with specified n and dim
            auto result3 = torch::fft::fft(input_tensor, n, dim);
            
            // FFT with all parameters
            auto result4 = torch::fft::fft(input_tensor, n, dim, norm);
            
            // ifft (inverse FFT)
            auto ifft_result = torch::fft::ifft(input_tensor);
        } catch (const c10::Error &e) {
            // Expected for invalid tensor types/shapes
        }

        // Try real FFT variants
        try {
            if (input_tensor.is_floating_point() && input_tensor.dim() >= 1) {
                auto rfft_result = torch::fft::rfft(input_tensor);
                auto ihfft_result = torch::fft::ihfft(input_tensor);
            }
        } catch (const c10::Error &e) {
            // Expected for invalid inputs
        }
        
        // Try complex FFT variants
        try {
            if (input_tensor.is_complex() && input_tensor.dim() >= 1) {
                auto hfft_result = torch::fft::hfft(input_tensor);
                auto irfft_result = torch::fft::irfft(input_tensor);
            }
        } catch (const c10::Error &e) {
            // Expected for invalid inputs
        }
        
        // Try 2D FFT variants if tensor has at least 2 dimensions
        try {
            if (input_tensor.dim() >= 2) {
                auto fft2_result = torch::fft::fft2(input_tensor);
                auto ifft2_result = torch::fft::ifft2(input_tensor);
            }
        } catch (const c10::Error &e) {
            // Expected for invalid inputs
        }

        try {
            if (input_tensor.dim() >= 2 && input_tensor.is_floating_point()) {
                auto rfft2_result = torch::fft::rfft2(input_tensor);
            }
        } catch (const c10::Error &e) {
            // Expected for invalid inputs
        }
        
        try {
            if (input_tensor.dim() >= 2 && input_tensor.is_complex()) {
                auto irfft2_result = torch::fft::irfft2(input_tensor);
            }
        } catch (const c10::Error &e) {
            // Expected for invalid inputs
        }
        
        // Try n-dimensional FFT variants
        try {
            auto fftn_result = torch::fft::fftn(input_tensor);
            auto ifftn_result = torch::fft::ifftn(input_tensor);
        } catch (const c10::Error &e) {
            // Expected for invalid inputs
        }
        
        try {
            if (input_tensor.is_floating_point()) {
                auto rfftn_result = torch::fft::rfftn(input_tensor);
            }
        } catch (const c10::Error &e) {
            // Expected for invalid inputs
        }
        
        try {
            if (input_tensor.is_complex()) {
                auto irfftn_result = torch::fft::irfftn(input_tensor);
            }
        } catch (const c10::Error &e) {
            // Expected for invalid inputs
        }
        
        // Try fftshift and ifftshift
        try {
            auto fftshift_result = torch::fft::fftshift(input_tensor);
            auto ifftshift_result = torch::fft::ifftshift(input_tensor);
        } catch (const c10::Error &e) {
            // Expected for invalid inputs
        }

        // Try fftfreq and rfftfreq
        try {
            if (n > 0) {
                auto options = torch::TensorOptions().dtype(torch::kFloat64);
                auto fftfreq_result = torch::fft::fftfreq(n, options);
                auto rfftfreq_result = torch::fft::rfftfreq(n, options);
            }
        } catch (const c10::Error &e) {
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