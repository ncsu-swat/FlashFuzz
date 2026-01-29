#include "fuzzer_utils.h" // General fuzzing utilities
#include <algorithm>      // For std::max
#include <cmath>          // For std::abs
#include <cstring>        // For std::memcpy
#include <iostream>       // For cerr
#include <optional>       // For std::optional
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor - ihfftn expects real input
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a real-valued floating point tensor
        if (!input_tensor.is_floating_point()) {
            input_tensor = input_tensor.to(torch::kFloat32);
        }
        
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
        
        // Normalize dimensions to valid range
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
        
        // Remove duplicate dimensions
        std::sort(dim_vec.begin(), dim_vec.end());
        dim_vec.erase(std::unique(dim_vec.begin(), dim_vec.end()), dim_vec.end());
        
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
                // Ensure s has same length as dim
                while (s_vec.size() > dim_ref.size()) {
                    s_vec.pop_back();
                }
                while (s_vec.size() < dim_ref.size()) {
                    s_vec.push_back(s_vec.empty() ? 2 : s_vec.back());
                }
                s = c10::IntArrayRef(s_vec);
            }
        }
        
        // Parse norm parameter if we have more data
        c10::optional<c10::string_view> norm = c10::nullopt;
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++];
            if (norm_selector % 4 == 0) {
                norm = "backward";
            } else if (norm_selector % 4 == 1) {
                norm = "forward";
            } else if (norm_selector % 4 == 2) {
                norm = "ortho";
            }
            // else leave as nullopt for default behavior
        }
        
        // Apply the ihfftn operation with different parameter combinations
        torch::Tensor result;
        
        // Try different combinations of parameters
        if (offset < Size) {
            uint8_t param_selector = Data[offset++];
            
            try {
                switch (param_selector % 5) {
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
                        // All parameters with nullopt for s
                        result = torch::fft::ihfftn(input_tensor, c10::nullopt, dim_ref, norm);
                        break;
                    case 4:
                        // Just dims, no s
                        result = torch::fft::ihfftn(input_tensor, c10::nullopt, dim_ref);
                        break;
                }
            } catch (const c10::Error &e) {
                // Expected errors from invalid parameter combinations
                return 0;
            }
        } else {
            // Default case if we don't have enough data
            try {
                result = torch::fft::ihfftn(input_tensor);
            } catch (const c10::Error &e) {
                // Expected errors from invalid input
                return 0;
            }
        }
        
        // Perform some operation on the result to ensure it's used
        // ihfftn returns complex tensor, use abs() before sum for real value
        auto sum = torch::abs(result).sum();
        
        // Prevent compiler from optimizing away the computation
        volatile float check = sum.item<float>();
        (void)check;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}