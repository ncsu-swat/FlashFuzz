#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a normalized dimension value for the FFT operation
        int64_t dim = -1;
        if (offset < Size) {
            // Extract a dimension value from the input data
            int64_t raw_dim;
            size_t bytes_to_read = std::min(sizeof(int64_t), Size - offset);
            std::memcpy(&raw_dim, Data + offset, bytes_to_read);
            offset += bytes_to_read;
            
            // If tensor has dimensions, select one of them
            if (input_tensor.dim() > 0) {
                dim = std::abs(raw_dim) % input_tensor.dim();
            }
        }
        
        // Get a normalized n value for the FFT operation
        int64_t n = -1;
        if (offset < Size) {
            // Extract an n value from the input data
            int64_t raw_n;
            size_t bytes_to_read = std::min(sizeof(int64_t), Size - offset);
            std::memcpy(&raw_n, Data + offset, bytes_to_read);
            offset += bytes_to_read;
            
            // Use raw_n as is, allowing negative values to test error handling
            n = raw_n;
        }
        
        // Get a normalized norm value for the FFT operation
        std::string norm = "backward";
        if (offset < Size) {
            // Use a byte to select normalization mode
            uint8_t norm_selector = Data[offset++];
            switch (norm_selector % 3) {
                case 0: norm = "backward"; break;
                case 1: norm = "ortho"; break;
                case 2: norm = "forward"; break;
            }
        }
        
        // Apply torch.fft.fft operation
        torch::Tensor result;
        
        // Try different variants of the FFT operation based on available parameters
        if (dim >= 0 && n >= 0) {
            // Full parameter version
            result = torch::fft::fft(input_tensor, n, dim, norm);
        } else if (dim >= 0) {
            // Without n
            result = torch::fft::fft(input_tensor, c10::nullopt, dim, norm);
        } else if (n >= 0) {
            // Without dim
            result = torch::fft::fft(input_tensor, n, -1, norm);
        } else {
            // Basic version
            result = torch::fft::fft(input_tensor);
        }
        
        // Try other FFT variants if there's more data
        if (offset < Size) {
            uint8_t fft_variant = Data[offset++];
            
            switch (fft_variant % 6) {
                case 0:
                    // ifft - inverse FFT
                    result = torch::fft::ifft(input_tensor, n, dim, norm);
                    break;
                case 1:
                    // rfft - real FFT
                    result = torch::fft::rfft(input_tensor, n, dim, norm);
                    break;
                case 2:
                    // irfft - inverse real FFT
                    result = torch::fft::irfft(input_tensor, n, dim, norm);
                    break;
                case 3:
                    // hfft - Hermitian FFT
                    result = torch::fft::hfft(input_tensor, n, dim, norm);
                    break;
                case 4:
                    // ihfft - inverse Hermitian FFT
                    result = torch::fft::ihfft(input_tensor, n, dim, norm);
                    break;
                case 5:
                    // fftfreq - FFT frequency bins
                    if (n > 0) {
                        double sample_spacing = 1.0;
                        if (offset < Size) {
                            // Extract sample spacing from input data
                            std::memcpy(&sample_spacing, Data + offset, std::min(sizeof(double), Size - offset));
                        }
                        result = torch::fft::fftfreq(n, sample_spacing);
                    }
                    break;
            }
        }
        
        // Try 2D FFT operations if tensor has at least 2 dimensions
        if (input_tensor.dim() >= 2 && offset < Size) {
            uint8_t fft2_variant = Data[offset++];
            
            // Extract dimensions for 2D FFT
            std::vector<int64_t> dim_vector;
            if (input_tensor.dim() >= 2) {
                dim_vector.push_back(0);
                dim_vector.push_back(1);
            }
            
            // Extract n values for 2D FFT
            std::vector<int64_t> n_vector;
            if (offset + sizeof(int64_t) * 2 <= Size) {
                int64_t n_values[2];
                std::memcpy(n_values, Data + offset, sizeof(int64_t) * 2);
                offset += sizeof(int64_t) * 2;
                
                // Use n values as is, allowing negative values to test error handling
                n_vector.push_back(n_values[0]);
                n_vector.push_back(n_values[1]);
            }
            
            switch (fft2_variant % 3) {
                case 0:
                    // fft2 - 2D FFT
                    if (!n_vector.empty()) {
                        result = torch::fft::fft2(input_tensor, n_vector, dim_vector, norm);
                    } else {
                        result = torch::fft::fft2(input_tensor, c10::nullopt, dim_vector, norm);
                    }
                    break;
                case 1:
                    // ifft2 - inverse 2D FFT
                    if (!n_vector.empty()) {
                        result = torch::fft::ifft2(input_tensor, n_vector, dim_vector, norm);
                    } else {
                        result = torch::fft::ifft2(input_tensor, c10::nullopt, dim_vector, norm);
                    }
                    break;
                case 2:
                    // rfft2 - real 2D FFT
                    if (!n_vector.empty()) {
                        result = torch::fft::rfft2(input_tensor, n_vector, dim_vector, norm);
                    } else {
                        result = torch::fft::rfft2(input_tensor, c10::nullopt, dim_vector, norm);
                    }
                    break;
            }
        }
        
        // Try n-dimensional FFT operations
        if (offset < Size) {
            uint8_t fftn_variant = Data[offset++];
            
            // Create a vector of dimensions for n-dimensional FFT
            std::vector<int64_t> dim_vector;
            int max_dims = std::min(static_cast<int>(input_tensor.dim()), 3);
            for (int i = 0; i < max_dims; i++) {
                dim_vector.push_back(i);
            }
            
            // Create a vector of n values for n-dimensional FFT
            std::vector<int64_t> n_vector;
            for (int i = 0; i < max_dims && offset + sizeof(int64_t) <= Size; i++) {
                int64_t n_value;
                std::memcpy(&n_value, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                n_vector.push_back(n_value);
            }
            
            switch (fftn_variant % 3) {
                case 0:
                    // fftn - n-dimensional FFT
                    if (!n_vector.empty()) {
                        result = torch::fft::fftn(input_tensor, n_vector, dim_vector, norm);
                    } else {
                        result = torch::fft::fftn(input_tensor, c10::nullopt, dim_vector, norm);
                    }
                    break;
                case 1:
                    // ifftn - inverse n-dimensional FFT
                    if (!n_vector.empty()) {
                        result = torch::fft::ifftn(input_tensor, n_vector, dim_vector, norm);
                    } else {
                        result = torch::fft::ifftn(input_tensor, c10::nullopt, dim_vector, norm);
                    }
                    break;
                case 2:
                    // rfftn - real n-dimensional FFT
                    if (!n_vector.empty()) {
                        result = torch::fft::rfftn(input_tensor, n_vector, dim_vector, norm);
                    } else {
                        result = torch::fft::rfftn(input_tensor, c10::nullopt, dim_vector, norm);
                    }
                    break;
            }
        }
        
        // Ensure result is used to prevent optimization
        auto sum = result.sum().item<double>();
        if (std::isnan(sum) || std::isinf(sum)) {
            // This is not an error, just a way to use the result
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}