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
        std::string norm_str = "backward";
        
        // Parse s (output shape) if we have enough data
        if (offset + 16 <= Size) {
            std::memcpy(&s_h, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            std::memcpy(&s_w, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
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
            switch (norm_selector % 4) {
                case 0: norm_str = "backward"; break;
                case 1: norm_str = "forward"; break;
                case 2: norm_str = "ortho"; break;
                default: norm_str = "backward"; break;
            }
        }
        
        // Create optional parameters
        at::OptionalIntArrayRef s = c10::nullopt;
        at::IntArrayRef dim = {-2, -1};
        
        // Set s if valid values were parsed
        if (s_h > 0 && s_w > 0) {
            std::vector<int64_t> s_vec{s_h, s_w};
            s = s_vec;
        }
        
        // Set dim if valid values were parsed
        if (dim_h >= 0 && dim_w >= 0 && dim_h != dim_w) {
            std::vector<int64_t> dim_vec{dim_h % input.dim(), dim_w % input.dim()};
            dim = dim_vec;
        }
        
        // Apply the hfft2 operation
        torch::Tensor output;
        
        // Try different combinations of optional parameters
        if (offset % 4 == 0) {
            // Call with all parameters
            output = torch::fft::hfft2(input, s, dim, norm_str);
        } else if (offset % 4 == 1) {
            // Call without s
            output = torch::fft::hfft2(input, c10::nullopt, dim, norm_str);
        } else if (offset % 4 == 2) {
            // Call without dim
            output = torch::fft::hfft2(input, s, {-2, -1}, norm_str);
        } else {
            // Call with minimal parameters
            output = torch::fft::hfft2(input);
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
