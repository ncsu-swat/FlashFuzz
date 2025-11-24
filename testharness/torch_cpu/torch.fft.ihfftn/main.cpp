#include "fuzzer_utils.h" // General fuzzing utilities
#include <algorithm>      // For std::max
#include <cmath>          // For std::abs
#include <cstring>        // For std::memcpy
#include <iostream>       // For cerr
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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse n_dims parameter if we have more data
        std::vector<int64_t> dim_vec;
        if (offset + 1 < Size) {
            uint8_t n_dims_count = Data[offset++] % 5; // Get up to 4 dimensions
            
            for (uint8_t i = 0; i < n_dims_count && offset < Size; i++) {
                if (offset + sizeof(int64_t) <= Size) {
                    int64_t dim_value;
                    std::memcpy(&dim_value, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    dim_vec.push_back(dim_value);
                } else {
                    // Not enough data for a full int64_t, use a default value
                    dim_vec.push_back(i + 1);
                }
            }
        }
        int64_t input_rank = input_tensor.dim();
        for (auto &d : dim_vec) {
            if (input_rank > 0) {
                int64_t wrapped = d % input_rank;
                if (wrapped < 0) {
                    wrapped += input_rank;
                }
                d = wrapped;
            } else {
                d = 0;
            }
        }
        std::vector<int64_t> default_dim;
        if (input_rank >= 2) {
            default_dim = {-2, -1};
        } else if (input_rank == 1) {
            default_dim = {0};
        }
        c10::IntArrayRef dim_ref = !dim_vec.empty() ? c10::IntArrayRef(dim_vec) : c10::IntArrayRef(default_dim);
        
        // Parse s parameter if we have more data
        c10::optional<c10::IntArrayRef> s = c10::nullopt;
        std::vector<int64_t> s_vec;
        if (offset + 1 < Size) {
            uint8_t s_count = Data[offset++] % 5; // Get up to 4 dimensions for s
            
            for (uint8_t i = 0; i < s_count && offset < Size; i++) {
                if (offset + sizeof(int64_t) <= Size) {
                    int64_t s_value;
                    std::memcpy(&s_value, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    s_vec.push_back(s_value);
                } else {
                    // Not enough data for a full int64_t, use a default value
                    s_vec.push_back(i + 1);
                }
            }
            if (!s_vec.empty()) {
                for (auto &val : s_vec) {
                    val = std::max<int64_t>(1, static_cast<int64_t>(std::abs(val) % 16) + 1);
                }
                s = c10::IntArrayRef(s_vec);
            }
        }
        
        // Parse norm parameter if we have more data
        std::optional<std::string_view> norm = std::nullopt;
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++];
            if (norm_selector % 3 == 0) {
                norm = "backward";
            } else if (norm_selector % 3 == 1) {
                norm = "forward";
            } else {
                norm = "ortho";
            }
        }
        
        // Apply the ihfftn operation with different parameter combinations
        torch::Tensor result;
        
        // Try different combinations of parameters
        if (offset < Size) {
            uint8_t param_selector = Data[offset++];
            
            switch (param_selector % 4) {
                case 0:
                    // Just input tensor
                    result = torch::fft::ihfftn(input_tensor);
                    break;
                case 1:
                    // Input tensor with optional shape and dims
                    result = torch::fft::ihfftn(input_tensor, s, dim_ref);
                    break;
                case 2:
                    // Input tensor, dim, and norm
                    result = torch::fft::ihfftn(input_tensor, s, dim_ref, norm);
                    break;
                case 3:
                    // All parameters
                    result = torch::fft::ihfftn(input_tensor, c10::nullopt, dim_ref, norm);
                    break;
            }
        } else {
            // Default case if we don't have enough data
            result = torch::fft::ihfftn(input_tensor);
        }
        
        // Perform some operation on the result to ensure it's used
        auto sum = result.sum();
        
        // Prevent compiler from optimizing away the computation
        if (sum.item<double>() == -12345.6789) {
            return 1; // This condition is unlikely to be true
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
