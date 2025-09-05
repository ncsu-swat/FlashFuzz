#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal bytes for configuration
        if (Size < 4) {
            return 0;  // Not enough data, but keep for coverage
        }

        // Create input tensor
        torch::Tensor input_tensor;
        try {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If tensor creation fails, try with minimal tensor
            input_tensor = torch::randn({1});
        }

        // Ensure we have a floating point type for nanmean
        if (!input_tensor.is_floating_point()) {
            // Convert to float to support NaN operations
            input_tensor = input_tensor.to(torch::kFloat32);
        }

        // Inject some NaN values based on fuzzer input
        if (offset < Size) {
            uint8_t nan_pattern = Data[offset++];
            if (nan_pattern % 3 == 0 && input_tensor.numel() > 0) {
                // Add NaN values at random positions
                auto flat = input_tensor.flatten();
                int64_t num_nans = (nan_pattern % flat.numel()) + 1;
                for (int64_t i = 0; i < num_nans && i < flat.numel(); ++i) {
                    int64_t idx = (nan_pattern * (i + 1)) % flat.numel();
                    flat[idx] = std::numeric_limits<float>::quiet_NaN();
                }
            } else if (nan_pattern % 3 == 1) {
                // Make entire tensor NaN
                input_tensor.fill_(std::numeric_limits<float>::quiet_NaN());
            }
            // else: keep original values (no NaN)
        }

        // Parse reduction dimensions
        std::vector<int64_t> dims;
        bool has_dims = false;
        if (offset < Size) {
            uint8_t use_dims = Data[offset++];
            if (use_dims % 2 == 0 && input_tensor.dim() > 0) {
                has_dims = true;
                if (offset < Size) {
                    uint8_t num_dims = (Data[offset++] % input_tensor.dim()) + 1;
                    for (uint8_t i = 0; i < num_dims && offset < Size; ++i) {
                        int64_t dim = Data[offset++] % input_tensor.dim();
                        // Avoid duplicates
                        if (std::find(dims.begin(), dims.end(), dim) == dims.end()) {
                            dims.push_back(dim);
                        }
                    }
                }
                if (dims.empty() && input_tensor.dim() > 0) {
                    dims.push_back(0);  // Default to first dimension
                }
            }
        }

        // Parse keepdim option
        bool keepdim = false;
        if (offset < Size) {
            keepdim = (Data[offset++] % 2 == 0);
        }

        // Test 1: Basic nanmean without dimensions
        try {
            torch::Tensor result1 = torch::nanmean(input_tensor);
            
            // Verify result properties
            if (result1.numel() != 1) {
                std::cerr << "Unexpected: nanmean without dims should return scalar" << std::endl;
            }
            
            // Check if result is NaN when all values are NaN
            if (input_tensor.isnan().all().item<bool>() && !result1.isnan().item<bool>()) {
                std::cerr << "Unexpected: all NaN input should give NaN result" << std::endl;
            }
        } catch (const c10::Error& e) {
            // Some tensor configurations might not support nanmean
            // Continue to test other paths
        }

        // Test 2: nanmean with specific dimensions
        if (has_dims && !dims.empty()) {
            try {
                torch::Tensor result2 = torch::nanmean(input_tensor, dims, keepdim);
                
                // Verify shape
                if (keepdim) {
                    if (result2.dim() != input_tensor.dim()) {
                        std::cerr << "Unexpected: keepdim should preserve dimensionality" << std::endl;
                    }
                } else {
                    int64_t expected_dim = input_tensor.dim() - dims.size();
                    if (expected_dim < 0) expected_dim = 0;
                    if (result2.dim() != expected_dim && result2.dim() != 0) {
                        std::cerr << "Unexpected dimension after reduction" << std::endl;
                    }
                }
            } catch (const c10::Error& e) {
                // Some dimension configurations might be invalid
                // Continue testing
            }
        }

        // Test 3: Edge case - empty tensor
        if (input_tensor.numel() == 0) {
            try {
                torch::Tensor empty_result = torch::nanmean(input_tensor);
                // Empty tensor nanmean behavior
            } catch (const c10::Error& e) {
                // Expected for some empty configurations
            }
        }

        // Test 4: Single dimension reduction with different keepdim values
        if (input_tensor.dim() > 0) {
            for (int64_t d = 0; d < input_tensor.dim(); ++d) {
                try {
                    torch::Tensor result_keep = torch::nanmean(input_tensor, {d}, true);
                    torch::Tensor result_no_keep = torch::nanmean(input_tensor, {d}, false);
                    
                    // Verify keepdim effect
                    if (result_keep.dim() != input_tensor.dim()) {
                        std::cerr << "Keepdim=true failed to preserve dimensions" << std::endl;
                    }
                } catch (const c10::Error& e) {
                    // Continue on errors
                }
            }
        }

        // Test 5: All dimensions reduction
        if (input_tensor.dim() > 1) {
            try {
                std::vector<int64_t> all_dims;
                for (int64_t i = 0; i < input_tensor.dim(); ++i) {
                    all_dims.push_back(i);
                }
                torch::Tensor all_reduce = torch::nanmean(input_tensor, all_dims, keepdim);
                
                if (!keepdim && all_reduce.dim() != 0) {
                    std::cerr << "Reducing all dims without keepdim should give scalar" << std::endl;
                }
            } catch (const c10::Error& e) {
                // Continue testing
            }
        }

        // Test 6: Mixed NaN patterns
        if (input_tensor.numel() > 2) {
            try {
                // Create a tensor with specific NaN pattern
                auto mixed = input_tensor.clone();
                mixed.flatten()[0] = std::numeric_limits<float>::quiet_NaN();
                torch::Tensor mixed_result = torch::nanmean(mixed);
                
                // The result should ignore the NaN value
                if (mixed_result.isnan().item<bool>() && !mixed.isnan().all().item<bool>()) {
                    std::cerr << "Unexpected NaN in result when input has non-NaN values" << std::endl;
                }
            } catch (const c10::Error& e) {
                // Continue
            }
        }

        // Test 7: Special values (inf, -inf)
        if (offset < Size && input_tensor.numel() > 0) {
            uint8_t special_val = Data[offset++];
            auto special_tensor = input_tensor.clone();
            if (special_val % 3 == 0) {
                special_tensor.flatten()[0] = std::numeric_limits<float>::infinity();
            } else if (special_val % 3 == 1) {
                special_tensor.flatten()[0] = -std::numeric_limits<float>::infinity();
            }
            
            try {
                torch::Tensor special_result = torch::nanmean(special_tensor);
                // Inf values should be included in mean (not treated as NaN)
            } catch (const c10::Error& e) {
                // Continue
            }
        }

        // Test 8: Negative dimensions
        if (input_tensor.dim() > 0 && offset < Size) {
            try {
                int64_t neg_dim = -(Data[offset++] % input_tensor.dim()) - 1;
                torch::Tensor neg_result = torch::nanmean(input_tensor, {neg_dim}, keepdim);
                // Negative indexing should work
            } catch (const c10::Error& e) {
                // Invalid negative dimension
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}