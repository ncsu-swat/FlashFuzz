#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test various torch functions
        try {
            // Test window functions using torch functions
            if (offset + 1 < Size) {
                uint8_t window_type = Data[offset++];
                int64_t window_length = 0;
                
                // Extract window length from input tensor if possible
                if (input.dim() > 0 && input.size(0) > 0) {
                    window_length = input.size(0);
                } else {
                    // Use a small default value if tensor is empty
                    window_length = 10;
                }
                
                // Limit window length to avoid excessive memory usage
                window_length = std::min(window_length, static_cast<int64_t>(1024));
                
                // Apply different window functions based on the extracted byte
                switch (window_type % 7) {
                    case 0:
                        torch::bartlett_window(window_length);
                        break;
                    case 1:
                        torch::blackman_window(window_length);
                        break;
                    case 2:
                        torch::hamming_window(window_length);
                        break;
                    case 3:
                        torch::hann_window(window_length);
                        break;
                    case 4: {
                        bool periodic = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
                        torch::kaiser_window(window_length, periodic, 0.5);
                        break;
                    }
                    case 5: {
                        double std_val = (offset < Size) ? (Data[offset++] % 100) / 100.0 : 0.5;
                        torch::gaussian_window(window_length, std_val);
                        break;
                    }
                    case 6: {
                        torch::hann_window(window_length);
                        break;
                    }
                }
            }
            
            // Test FFT functions
            if (input.dim() > 0) {
                try {
                    torch::Tensor fft_result = torch::fft::fft(input);
                    torch::Tensor ifft_result = torch::fft::ifft(fft_result);
                } catch (...) {
                    // FFT may fail for certain input shapes/types, continue with other tests
                }
                
                // Test rfft if input is real
                if (torch::isFloatingType(input.scalar_type()) || 
                    input.scalar_type() == torch::kInt || 
                    input.scalar_type() == torch::kLong) {
                    try {
                        torch::Tensor rfft_result = torch::fft::rfft(input.to(torch::kFloat));
                        torch::Tensor irfft_result = torch::fft::irfft(rfft_result);
                    } catch (...) {
                        // RFFT may fail for certain inputs, continue
                    }
                }
                
                // Test 2D FFT if tensor has at least 2 dimensions
                if (input.dim() >= 2) {
                    try {
                        torch::Tensor fft2_result = torch::fft::fft2(input);
                        torch::Tensor ifft2_result = torch::fft::ifft2(fft2_result);
                    } catch (...) {
                        // 2D FFT may fail for certain inputs, continue
                    }
                }
            }
            
            // Test stft if input is 1D
            if (input.dim() == 1 && input.numel() > 0) {
                try {
                    int64_t n_fft = 16;
                    int64_t hop_length = 4;
                    torch::Tensor stft_result = torch::stft(
                        input.to(torch::kFloat),
                        n_fft,
                        hop_length,
                        n_fft,
                        torch::hann_window(n_fft),
                        /*center=*/true,
                        /*normalized=*/false,
                        /*onesided=*/true,
                        /*return_complex=*/true
                    );
                } catch (...) {
                    // STFT may fail for certain inputs, continue
                }
            }
            
            // Test convolution functions
            if (offset + 1 < Size && input.dim() > 0) {
                try {
                    // Create a small kernel
                    std::vector<int64_t> kernel_shape;
                    for (int i = 0; i < input.dim(); i++) {
                        kernel_shape.push_back(std::min(static_cast<int64_t>(3), input.size(i)));
                    }
                    
                    torch::Tensor kernel = torch::ones(kernel_shape, input.options());
                    
                    // Test different convolution modes using conv1d
                    if (input.dim() == 1) {
                        torch::Tensor input_3d = input.unsqueeze(0).unsqueeze(0);
                        torch::Tensor kernel_3d = kernel.unsqueeze(0).unsqueeze(0);
                        torch::Tensor conv_result = torch::conv1d(input_3d, kernel_3d);
                    }
                } catch (...) {
                    // Convolution may fail for certain inputs, continue
                }
            }
            
            // Test correlation functions using cross-correlation
            if (offset + 1 < Size && input.dim() > 0) {
                try {
                    // Create a small kernel
                    std::vector<int64_t> kernel_shape;
                    for (int i = 0; i < input.dim(); i++) {
                        kernel_shape.push_back(std::min(static_cast<int64_t>(3), input.size(i)));
                    }
                    
                    torch::Tensor kernel = torch::ones(kernel_shape, input.options());
                    
                    // Test correlation using convolution with flipped kernel
                    if (input.dim() == 1) {
                        torch::Tensor input_3d = input.unsqueeze(0).unsqueeze(0);
                        torch::Tensor kernel_3d = kernel.flip({0}).unsqueeze(0).unsqueeze(0);
                        torch::Tensor corr_result = torch::conv1d(input_3d, kernel_3d);
                    }
                } catch (...) {
                    // Correlation may fail for certain inputs, continue
                }
            }
        } catch (const std::exception &e) {
            // Catch exceptions from signal operations but continue with other tests
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}