#include "fuzzer_utils.h" // General fuzzing utilities
#include <cstring>        // For std::memcpy
#include <iostream>       // For cerr
#include <cstdlib>        // For std::abs
#include <optional>       // For std::optional
#include <string_view>    // For std::string_view
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor and specify FFT parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least one more byte for FFT parameters
        if (offset >= Size) {
            return 0;
        }
        
        // Parse FFT parameters
        uint8_t fft_param_byte = Data[offset++];
        
        // Extract FFT dimension parameter
        int64_t dim = -1;
        if (input_tensor.dim() > 0) {
            dim = fft_param_byte % input_tensor.dim();
        }
        
        // Extract FFT normalization parameter
        uint8_t norm_selector = 0;
        if (offset < Size) {
            norm_selector = Data[offset++];
        }
        
        // Select normalization mode
        std::string norm;
        std::optional<std::string_view> norm_opt;
        switch (norm_selector % 4) {
            case 0:
                norm = "forward";
                break;
            case 1:
                norm = "backward";
                break;
            case 2:
                norm = "ortho";
                break;
            default:
                norm = ""; // Empty string for default behavior
        }
        if (!norm.empty()) {
            norm_opt = std::string_view(norm);
        }
        
        // Extract n parameter (optional)
        int64_t n_raw = -1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&n_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        std::optional<torch::SymInt> n_opt;
        if (n_raw != 0) {
            int64_t n = 1 + (std::abs(n_raw) % 64);
            n_opt = torch::SymInt(n);
        }
        
        // Try different FFT operations based on available data
        try {
            // Basic FFT (1D)
            if (input_tensor.dim() > 0) {
                torch::Tensor result1 = torch::fft::fft(input_tensor);
            }
            
            // FFT with dimension specified
            if (input_tensor.dim() > 0) {
                torch::Tensor result2 = torch::fft::fft(input_tensor, std::nullopt, dim);
            }
            
            // FFT with normalization
            if (norm_opt) {
                torch::Tensor result3 = torch::fft::fft(input_tensor, std::nullopt, dim, norm_opt);
            }
            
            // FFT with n parameter
            if (n_opt) {
                torch::Tensor result4 = torch::fft::fft(input_tensor, n_opt, -1, std::string_view("forward"));
            }
            
            // FFT with all parameters
            if (input_tensor.dim() > 0 && norm_opt && n_opt) {
                torch::Tensor result5 = torch::fft::fft(input_tensor, n_opt, dim, norm_opt);
            }
            
            // Try other FFT variants if we have enough dimensions
            if (input_tensor.dim() >= 2) {
                // 2D FFT
                torch::Tensor result6 = torch::fft::fft2(input_tensor);
                
                // N-dimensional FFT
                torch::Tensor result7 = torch::fft::fftn(input_tensor);
                
                // IFFT (inverse FFT)
                torch::Tensor result8 = torch::fft::ifft(input_tensor);
                
                // 2D IFFT
                torch::Tensor result9 = torch::fft::ifft2(input_tensor);
                
                // N-dimensional IFFT
                torch::Tensor result10 = torch::fft::ifftn(input_tensor);
            }
            
            // RFFT (real FFT) - works on real inputs
            if (input_tensor.is_floating_point() && !input_tensor.is_complex()) {
                torch::Tensor result11 = torch::fft::rfft(input_tensor);
                
                if (input_tensor.dim() >= 2) {
                    // 2D RFFT
                    torch::Tensor result12 = torch::fft::rfft2(input_tensor);
                    
                    // N-dimensional RFFT
                    torch::Tensor result13 = torch::fft::rfftn(input_tensor);
                }
            }
            
            // IRFFT (inverse real FFT) - works on complex inputs
            if (input_tensor.is_complex()) {
                torch::Tensor result14 = torch::fft::irfft(input_tensor);
                
                if (input_tensor.dim() >= 2) {
                    // 2D IRFFT
                    torch::Tensor result15 = torch::fft::irfft2(input_tensor);
                    
                    // N-dimensional IRFFT
                    torch::Tensor result16 = torch::fft::irfftn(input_tensor);
                }
            }
            
            // HFFT and IHFFT
            if (input_tensor.is_complex()) {
                torch::Tensor result17 = torch::fft::hfft(input_tensor);
            }
            
            if (input_tensor.is_floating_point() && !input_tensor.is_complex()) {
                torch::Tensor result18 = torch::fft::ihfft(input_tensor);
            }
            
            // FFT shift operations
            torch::Tensor result19 = torch::fft::fftshift(input_tensor);
            torch::Tensor result20 = torch::fft::ifftshift(input_tensor);
            
            // Try with specific dimensions for shift operations
            if (input_tensor.dim() > 0) {
                std::vector<int64_t> dims = {dim};
                torch::Tensor result21 = torch::fft::fftshift(input_tensor, dims);
                torch::Tensor result22 = torch::fft::ifftshift(input_tensor, dims);
            }
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected and part of testing
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
