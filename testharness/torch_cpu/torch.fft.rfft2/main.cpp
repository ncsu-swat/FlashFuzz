#include "fuzzer_utils.h" // General fuzzing utilities
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>       // For cerr
#include <optional>
#include <string_view>
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
        
        // Parse optional parameters if we have more data
        int64_t s = -1;                // Default: use input size
        int64_t dim1 = -1, dim2 = -1;  // Default dimensions
        std::optional<std::string_view> norm = std::nullopt;
        
        // Parse dimensions if we have more data
        if (offset + 2 < Size) {
            // Get dimensions for rfft2
            uint8_t dim_selector1 = Data[offset++];
            uint8_t dim_selector2 = Data[offset++];
            
            // Allow negative dimensions to test error handling
            if (input.dim() > 0) {
                dim1 = static_cast<int8_t>(dim_selector1) % (2 * input.dim()) - input.dim();
                dim2 = static_cast<int8_t>(dim_selector2) % (2 * input.dim()) - input.dim();
            }
        }
        
        // Parse s parameter if we have more data
        if (offset + sizeof(int32_t) <= Size) {
            int32_t s_raw;
            std::memcpy(&s_raw, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            
            // Bound positive sizes to keep allocations in check
            if (s_raw > 0)
            {
                constexpr int64_t max_fft_size = 16;
                s = std::max<int64_t>(1, std::min<int64_t>(s_raw, max_fft_size));
            }
            else
            {
                s = s_raw;
            }
        }
        
        // Parse norm parameter if we have more data
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++];
            switch (norm_selector % 4) {
                case 0:
                    norm = "backward";
                    break;
                case 1:
                    norm = "forward";
                    break;
                case 2:
                    norm = "ortho";
                    break;
                default:
                    norm = std::nullopt;
                    break;
            }
        }
        
        // Create a vector of dimensions
        std::vector<int64_t> dim;
        if (dim1 >= 0 || dim2 >= 0) {
            if (dim1 >= 0) dim.push_back(dim1);
            if (dim2 >= 0) dim.push_back(dim2);
        } else {
            // Default: use last 2 dimensions or fewer if tensor doesn't have 2 dims
            int64_t ndim = input.dim();
            if (ndim >= 2) {
                dim.push_back(ndim - 2);
                dim.push_back(ndim - 1);
            } else if (ndim == 1) {
                dim.push_back(0);
            }
        }
        
        // Prepare optional size overrides and dims (API defaults to {-2, -1})
        static const std::array<int64_t, 2> default_dims = {-2, -1};
        size_t target_dim_count = dim.empty() ? default_dims.size() : dim.size();

        // Create s parameter
        std::vector<int64_t> s_vec;
        if (s > 0)
        {
            // Use the provided s value for all dimensions
            for (size_t i = 0; i < target_dim_count; i++)
            {
                s_vec.push_back(s);
            }
        }

        c10::optional<at::IntArrayRef> s_opt = c10::nullopt;
        if (!s_vec.empty())
        {
            s_opt = at::IntArrayRef(s_vec);
        }

        at::IntArrayRef dim_ref = dim.empty() ? at::IntArrayRef(default_dims) : at::IntArrayRef(dim);
        
        // Apply rfft2 operation with different parameter combinations
        torch::Tensor output = torch::fft::rfft2(input, s_opt, dim_ref, norm);
        
        // Verify output is not empty
        if (output.numel() == 0 && input.numel() > 0) {
            throw std::runtime_error("rfft2 produced empty output for non-empty input");
        }
        
        // Try inverse operation to ensure roundtrip works
        torch::Tensor reconstructed = torch::fft::irfft2(output, s_opt, dim_ref, norm);
        
        // Try some additional operations on the output
        torch::Tensor abs_output = output.abs();
        torch::Tensor sum_output = output.sum();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
