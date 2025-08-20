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
        
        // Parse FFT parameters if we have more data
        int64_t n = -1;  // Default: use input size
        int64_t dim = -1; // Default: last dimension
        std::string norm_str;
        bool has_norm = false;
        
        // Parse n parameter if we have enough data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&n, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Parse dim parameter if we have enough data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // If tensor has dimensions, ensure dim is within valid range
            if (input_tensor.dim() > 0) {
                dim = dim % input_tensor.dim();
                if (dim < 0) {
                    dim += input_tensor.dim();
                }
            }
        }
        
        // Parse norm parameter if we have enough data
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++];
            if (norm_selector % 4 == 0) {
                norm_str = "forward";
                has_norm = true;
            } else if (norm_selector % 4 == 1) {
                norm_str = "backward";
                has_norm = true;
            } else if (norm_selector % 4 == 2) {
                norm_str = "ortho";
                has_norm = true;
            }
            // else leave norm as empty (default behavior)
        }
        
        // Apply FFT operation with different parameter combinations
        torch::Tensor result;
        
        // Case 1: Basic FFT with default parameters
        result = torch::fft::fft(input_tensor);
        
        // Case 2: FFT with specified n
        if (n != -1) {
            result = torch::fft::fft(input_tensor, n);
        }
        
        // Case 3: FFT with specified n and dim
        if (n != -1 && dim != -1) {
            result = torch::fft::fft(input_tensor, n, dim);
        }
        
        // Case 4: FFT with all parameters
        if (n != -1 && dim != -1 && has_norm) {
            result = torch::fft::fft(input_tensor, n, dim, norm_str);
        }
        
        // Case 5: FFT with just dim
        if (dim != -1) {
            result = torch::fft::fft(input_tensor, c10::nullopt, dim);
        }
        
        // Case 6: FFT with just norm
        if (has_norm) {
            result = torch::fft::fft(input_tensor, c10::nullopt, -1, norm_str);
        }
        
        // Case 7: FFT with dim and norm
        if (dim != -1 && has_norm) {
            result = torch::fft::fft(input_tensor, c10::nullopt, dim, norm_str);
        }
        
        // Case 8: FFT with n and norm
        if (n != -1 && has_norm) {
            result = torch::fft::fft(input_tensor, n, -1, norm_str);
        }
        
        // Access result elements to ensure computation is performed
        if (result.numel() > 0) {
            auto flat_result = result.flatten();
            auto first_element = flat_result[0];
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}