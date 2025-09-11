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
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse n_dims parameter if we have more data
        std::vector<int64_t> dim;
        if (offset + 1 < Size) {
            uint8_t n_dims = Data[offset++] % 5; // Get number of dimensions to transform
            
            // Parse dimensions to transform
            for (uint8_t i = 0; i < n_dims && offset < Size; i++) {
                int64_t d = static_cast<int64_t>(Data[offset++]);
                dim.push_back(d);
            }
        }
        
        // Parse s parameter (shape of the transformed axis)
        std::vector<int64_t> s;
        if (offset + 1 < Size) {
            uint8_t s_size = Data[offset++] % 5; // Get size of s parameter
            
            // Parse s values
            for (uint8_t i = 0; i < s_size && offset + sizeof(int64_t) <= Size; i++) {
                int64_t s_val;
                std::memcpy(&s_val, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                s.push_back(s_val);
            }
        }
        
        // Parse norm parameter
        std::optional<std::string> norm = std::nullopt;
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++] % 4;
            if (norm_selector == 0) {
                norm = "backward";
            } else if (norm_selector == 1) {
                norm = "forward";
            } else if (norm_selector == 2) {
                norm = "ortho";
            }
            // norm_selector == 3 keeps norm as std::nullopt
        }
        
        // Apply torch.fft.hfftn operation
        torch::Tensor result;
        
        // Call with different combinations of parameters based on available data
        if (dim.empty() && s.empty() && !norm) {
            result = torch::fft::hfftn(input_tensor);
        } else if (!dim.empty() && s.empty() && !norm) {
            result = torch::fft::hfftn(input_tensor, dim);
        } else if (dim.empty() && !s.empty() && !norm) {
            result = torch::fft::hfftn(input_tensor, {}, s);
        } else if (dim.empty() && s.empty() && norm) {
            result = torch::fft::hfftn(input_tensor, {}, {}, *norm);
        } else if (!dim.empty() && !s.empty() && !norm) {
            result = torch::fft::hfftn(input_tensor, dim, s);
        } else if (!dim.empty() && s.empty() && norm) {
            result = torch::fft::hfftn(input_tensor, dim, {}, *norm);
        } else if (dim.empty() && !s.empty() && norm) {
            result = torch::fft::hfftn(input_tensor, {}, s, *norm);
        } else {
            result = torch::fft::hfftn(input_tensor, dim, s, norm ? *norm : "backward");
        }
        
        // Access result to ensure computation is performed
        auto result_size = result.sizes();
        auto result_dtype = result.dtype();
        
        // Try to perform a simple operation on the result to ensure it's valid
        if (result.numel() > 0) {
            auto sum = result.sum();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
