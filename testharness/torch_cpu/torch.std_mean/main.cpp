#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Skip if not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor - needs to be floating point for std_mean
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor is floating point type (required for std_mean)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Extract correction parameter (0 = population std, 1 = sample std, etc.)
        int64_t correction = 1; // Default to Bessel's correction
        if (offset < Size) {
            correction = static_cast<int64_t>(Data[offset++] % 3); // 0, 1, or 2
        }
        
        // Get dimension parameter if there's data left
        bool keepdim = false;
        int64_t dim_value = 0;
        bool use_dim = false;
        
        if (offset < Size) {
            use_dim = Data[offset++] & 0x1;
        }
        
        if (use_dim && offset < Size && input.dim() > 0) {
            dim_value = static_cast<int64_t>(Data[offset++]);
            // Ensure dimension is valid
            dim_value = dim_value % input.dim();
            
            if (offset < Size) {
                keepdim = Data[offset++] & 0x1;
            }
        }
        
        // Variant 1: std_mean over all elements (no dimension specified)
        // Returns tuple of (std, mean)
        {
            auto result = torch::std_mean(input);
            auto std_val = std::get<0>(result);
            auto mean_val = std::get<1>(result);
            (void)std_val;
            (void)mean_val;
        }
        
        // Variant 2: std_mean with dimension specified
        if (use_dim && input.dim() > 0) {
            try {
                auto result = torch::std_mean(input, dim_value, correction, keepdim);
                auto std_val = std::get<0>(result);
                auto mean_val = std::get<1>(result);
                (void)std_val;
                (void)mean_val;
            } catch (...) {
                // Silently catch dimension-related errors
            }
        }
        
        // Variant 3: std_mean with dimension list
        if (input.dim() > 1 && offset < Size) {
            try {
                std::vector<int64_t> dims;
                int max_dims = std::min(static_cast<int64_t>(input.dim()), static_cast<int64_t>(2));
                
                for (int64_t i = 0; i < max_dims && offset < Size; i++) {
                    int64_t d = static_cast<int64_t>(Data[offset++]) % input.dim();
                    // Avoid duplicate dimensions
                    bool duplicate = false;
                    for (auto existing_d : dims) {
                        if (existing_d == d) {
                            duplicate = true;
                            break;
                        }
                    }
                    if (!duplicate) {
                        dims.push_back(d);
                    }
                }
                
                if (!dims.empty()) {
                    auto result = torch::std_mean(input, dims, correction, keepdim);
                    auto std_val = std::get<0>(result);
                    auto mean_val = std::get<1>(result);
                    (void)std_val;
                    (void)mean_val;
                }
            } catch (...) {
                // Silently catch dimension-related errors
            }
        }
        
        // Variant 4: std_mean with correction=0 (population std)
        {
            try {
                auto result = torch::std_mean(input, /*dim=*/0, /*correction=*/0, /*keepdim=*/false);
                auto std_val = std::get<0>(result);
                auto mean_val = std::get<1>(result);
                (void)std_val;
                (void)mean_val;
            } catch (...) {
                // Silently catch errors for edge cases
            }
        }
        
        // Variant 5: Test with different tensor configurations
        if (offset + 2 < Size && input.numel() > 0) {
            try {
                // Create a contiguous copy
                torch::Tensor contiguous_input = input.contiguous();
                auto result = torch::std_mean(contiguous_input);
                (void)std::get<0>(result);
                (void)std::get<1>(result);
            } catch (...) {
                // Silently catch errors
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}