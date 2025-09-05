#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 3) {
        // Need at least 3 bytes: 2 for tensor creation + 1 for N parameter
        return 0;
    }

    try {
        size_t offset = 0;
        
        // Create input tensor x
        torch::Tensor x = fuzzer_utils::createTensor(data, size, offset);
        
        // Ensure x has at least 1 dimension (vander expects at least 1D tensor)
        if (x.dim() == 0) {
            // Convert scalar to 1D tensor with single element
            x = x.unsqueeze(0);
        }
        
        // Parse optional N parameter
        c10::optional<int64_t> N;
        if (offset < size) {
            uint8_t n_selector = data[offset++];
            
            // Use different strategies based on selector value
            if (n_selector < 64) {
                // 25% chance: N is None (use default)
                N = c10::nullopt;
            } else if (n_selector < 192) {
                // 50% chance: N is a reasonable value
                if (offset + sizeof(int64_t) <= size) {
                    int64_t n_raw;
                    std::memcpy(&n_raw, data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    
                    // Bound N to reasonable range [1, 100] to avoid memory issues
                    n_raw = std::abs(n_raw);
                    N = 1 + (n_raw % 100);
                } else {
                    // Use small default if not enough data
                    N = 1 + (n_selector % 10);
                }
            } else {
                // 25% chance: edge cases
                uint8_t edge_case = n_selector % 4;
                switch (edge_case) {
                    case 0: N = 0; break;  // Zero columns
                    case 1: N = 1; break;  // Single column
                    case 2: N = x.size(-1); break;  // Same as input size
                    case 3: N = x.size(-1) + (n_selector % 5); break;  // Larger than input
                }
            }
        }
        
        // Test different dtype conversions if we have more data
        if (offset < size) {
            uint8_t dtype_convert = data[offset++];
            
            // Convert to different dtypes to test type support
            switch (dtype_convert % 8) {
                case 0: x = x.to(torch::kFloat32); break;
                case 1: x = x.to(torch::kFloat64); break;
                case 2: 
                    if (x.is_floating_point()) {
                        x = x.to(torch::kComplexFloat);
                    }
                    break;
                case 3:
                    if (x.is_floating_point()) {
                        x = x.to(torch::kComplexDouble);
                    }
                    break;
                case 4: x = x.to(torch::kInt32); break;
                case 5: x = x.to(torch::kInt64); break;
                case 6: x = x.to(torch::kInt8); break;
                case 7: x = x.to(torch::kInt16); break;
            }
        }
        
        // Call torch::linalg::vander
        torch::Tensor result;
        if (N.has_value()) {
            result = torch::linalg::vander(x, N.value());
        } else {
            result = torch::linalg::vander(x);
        }
        
        // Perform basic validation checks
        if (result.defined()) {
            // Check output shape
            auto input_shape = x.sizes();
            int64_t expected_rows = x.size(-1);
            int64_t expected_cols = N.has_value() ? N.value() : expected_rows;
            
            if (result.size(-2) != expected_rows && expected_cols > 0) {
                std::cerr << "Unexpected number of rows: " << result.size(-2) 
                         << " vs expected " << expected_rows << std::endl;
            }
            
            if (result.size(-1) != expected_cols) {
                std::cerr << "Unexpected number of columns: " << result.size(-1) 
                         << " vs expected " << expected_cols << std::endl;
            }
            
            // Check batch dimensions are preserved
            if (x.dim() > 1 && result.dim() != x.dim() + 1) {
                std::cerr << "Batch dimensions not preserved correctly" << std::endl;
            }
            
            // Test edge case operations
            if (offset < size && data[offset] % 4 == 0) {
                // Test flip operation (as mentioned in docs)
                auto flipped = result.flip(-1);
                
                // Test that result is finite for reasonable inputs
                if (x.is_floating_point() && x.numel() > 0) {
                    auto is_finite = torch::isfinite(result).all();
                    // Don't check the result - just ensure operation completes
                }
            }
            
            // Test with different memory layouts if we have more data
            if (offset < size && data[offset] % 3 == 0) {
                // Make input non-contiguous
                if (x.numel() > 1 && x.dim() > 0) {
                    x = x.transpose(0, -1).transpose(0, -1);
                    
                    // Call again with non-contiguous input
                    torch::Tensor result2;
                    if (N.has_value()) {
                        result2 = torch::linalg::vander(x, N.value());
                    } else {
                        result2 = torch::linalg::vander(x);
                    }
                }
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected for invalid inputs
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}