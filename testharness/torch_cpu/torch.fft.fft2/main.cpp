#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse FFT2 parameters if we have more data
        int64_t n_h = -1;
        int64_t n_w = -1;
        int64_t dim_h = -1;
        int64_t dim_w = -1;
        bool normalized = false;
        
        // Parse n_h and n_w (output size)
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&n_h, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&n_w, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Parse dim_h and dim_w (dimensions to transform)
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim_h, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim_w, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Parse normalized flag
        if (offset < Size) {
            normalized = Data[offset++] & 0x1;
        }
        
        // Apply FFT2 operation with different parameter combinations
        torch::Tensor output;
        
        // Try different combinations of parameters
        if (offset % 4 == 0) {
            // Case 1: Basic FFT2 with no parameters
            output = torch::fft::fft2(input);
        } 
        else if (offset % 4 == 1) {
            // Case 2: FFT2 with specified output size
            if (n_h > 0 && n_w > 0) {
                output = torch::fft::fft2(input, {n_h, n_w});
            } else {
                output = torch::fft::fft2(input);
            }
        }
        else if (offset % 4 == 2) {
            // Case 3: FFT2 with specified dimensions
            std::vector<int64_t> dims;
            if (dim_h >= -input.dim() && dim_h < input.dim()) {
                dims.push_back(dim_h);
                if (dim_w >= -input.dim() && dim_w < input.dim() && dim_w != dim_h) {
                    dims.push_back(dim_w);
                }
            }
            
            if (!dims.empty()) {
                std::optional<std::string_view> norm_opt = normalized ? std::optional<std::string_view>("ortho") : std::nullopt;
                output = torch::fft::fft2(input, c10::nullopt, dims, norm_opt);
            } else {
                std::optional<std::string_view> norm_opt = normalized ? std::optional<std::string_view>("ortho") : std::nullopt;
                output = torch::fft::fft2(input, c10::nullopt, {-2, -1}, norm_opt);
            }
        }
        else {
            // Case 4: FFT2 with all parameters
            std::vector<int64_t> dims;
            if (dim_h >= -input.dim() && dim_h < input.dim()) {
                dims.push_back(dim_h);
                if (dim_w >= -input.dim() && dim_w < input.dim() && dim_w != dim_h) {
                    dims.push_back(dim_w);
                }
            }
            
            std::optional<std::string_view> norm_opt = normalized ? std::optional<std::string_view>("ortho") : std::nullopt;
            
            if (n_h > 0 && n_w > 0 && !dims.empty()) {
                output = torch::fft::fft2(input, {n_h, n_w}, dims, norm_opt);
            } else if (n_h > 0 && n_w > 0) {
                output = torch::fft::fft2(input, {n_h, n_w}, {-2, -1}, norm_opt);
            } else if (!dims.empty()) {
                output = torch::fft::fft2(input, c10::nullopt, dims, norm_opt);
            } else {
                output = torch::fft::fft2(input, c10::nullopt, {-2, -1}, norm_opt);
            }
        }
        
        // Force evaluation of the output tensor
        auto sum = output.sum();
        
        // Try inverse FFT2 to check round-trip
        if (offset % 2 == 0) {
            auto inverse = torch::fft::ifft2(output);
            auto inverse_sum = inverse.sum();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
