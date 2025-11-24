#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <algorithm>      // For std::max
#include <string_view>    // For norm string handling
#include <optional>       // For std::optional norm handling

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse optional parameters if we have more data
        int64_t s_h = -1;
        int64_t s_w = -1;
        int64_t dim_h = -1;
        int64_t dim_w = -1;
        std::optional<std::string_view> norm = std::nullopt;
        
        // Parse s parameter (output size)
        if (offset + 2 < Size) {
            uint8_t use_s = Data[offset++] % 2;
            if (use_s) {
                if (offset + 2 * sizeof(int64_t) <= Size) {
                    memcpy(&s_h, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    memcpy(&s_w, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    
                    // Make sure s values are reasonable
                    s_h = std::max<int64_t>(1, std::abs(s_h) % 100);
                    s_w = std::max<int64_t>(1, std::abs(s_w) % 100);
                }
            }
        }
        
        // Parse dim parameter
        if (offset + 2 < Size) {
            uint8_t use_dim = Data[offset++] % 2;
            if (use_dim) {
                if (offset + 2 * sizeof(int64_t) <= Size) {
                    memcpy(&dim_h, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    memcpy(&dim_w, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    
                    // Allow negative dimensions for testing edge cases
                    dim_h = dim_h % 10;
                    dim_w = dim_w % 10;
                }
            }
        }
        
        // Parse norm parameter
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++] % 3;
            if (norm_selector == 1) {
                norm = "backward";
            } else if (norm_selector == 2) {
                norm = "ortho";
            }
        }
        
        // Apply the ihfft2 operation with different parameter combinations
        torch::Tensor output;
        
        // Case 1: Basic call with just the input tensor
        if (s_h < 0 && dim_h < 0) {
            if (norm) {
                output = torch::fft::ihfft2(input, c10::nullopt, {-2, -1}, norm);
            } else {
                output = torch::fft::ihfft2(input);
            }
        }
        // Case 2: With s parameter
        else if (s_h >= 0 && dim_h < 0) {
            std::vector<int64_t> s = {s_h, s_w};
            if (norm) {
                output = torch::fft::ihfft2(input, s, {-2, -1}, norm);
            } else {
                output = torch::fft::ihfft2(input, s);
            }
        }
        // Case 3: With dim parameter
        else if (s_h < 0 && dim_h >= 0) {
            std::vector<int64_t> dim = {dim_h, dim_w};
            if (norm) {
                output = torch::fft::ihfft2(input, c10::nullopt, dim, norm);
            } else {
                output = torch::fft::ihfft2(input, c10::nullopt, dim);
            }
        }
        // Case 4: With both s and dim parameters
        else if (s_h >= 0 && dim_h >= 0) {
            std::vector<int64_t> s = {s_h, s_w};
            std::vector<int64_t> dim = {dim_h, dim_w};
            if (norm) {
                output = torch::fft::ihfft2(input, s, dim, norm);
            } else {
                output = torch::fft::ihfft2(input, s, dim);
            }
        }
        
        // Perform some operation on the output to ensure it's used
        auto sum = output.sum();
        
        // Check if the output has NaN or Inf values
        if (sum.isnan().item<bool>() || sum.isinf().item<bool>()) {
            return 1; // Keep inputs that produce NaN or Inf
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
