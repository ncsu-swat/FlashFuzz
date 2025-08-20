#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for irfft2 from the remaining data
        std::vector<int64_t> s;
        int64_t dim_h = -2;
        int64_t dim_w = -1;
        int64_t norm_value = 0;
        
        // Parse s (output size)
        if (offset + 2 < Size) {
            uint8_t s_rank = Data[offset++] % 3; // 0, 1, or 2 dimensions for s
            
            for (uint8_t i = 0; i < s_rank && offset + sizeof(int64_t) <= Size; ++i) {
                int64_t dim_size;
                std::memcpy(&dim_size, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // Allow any dimension size including negative (to test error handling)
                s.push_back(dim_size);
            }
        }
        
        // Parse dim parameters
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim_h, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim_w, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Parse norm parameter
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
        
        // Apply irfft2 operation with various parameter combinations
        torch::Tensor output;
        
        // Try different combinations of parameters
        if (s.empty()) {
            if (dim_h == -2 && dim_w == -1) {
                // Basic case: no s, default dims
                output = torch::fft::irfft2(input, std::nullopt, {-2, -1}, norm);
            } else {
                // No s, with custom dims
                output = torch::fft::irfft2(input, std::nullopt, {dim_h, dim_w}, norm);
            }
        } else {
            // Convert s to IntArrayRef
            c10::IntArrayRef s_array(s.data(), s.size());
            
            if (dim_h == -2 && dim_w == -1) {
                // With s, default dims
                output = torch::fft::irfft2(input, s_array, {-2, -1}, norm);
            } else {
                // With s and custom dims
                output = torch::fft::irfft2(input, s_array, {dim_h, dim_w}, norm);
            }
        }
        
        // Perform some operation on the output to ensure it's used
        auto sum = output.sum();
        
        // Try to trigger potential issues with gradients
        if (offset < Size && Data[offset] % 2 == 0) {
            input = input.detach().requires_grad_(true);
            auto out = torch::fft::irfft2(input, std::nullopt, {-2, -1}, norm);
            auto sum_grad = out.sum();
            sum_grad.backward();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}