#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 4 bytes: 2 for tensors metadata, 1 for dim, 1 for eps selector
        if (Size < 4) {
            return 0;
        }

        // Create first tensor
        torch::Tensor x1;
        try {
            x1 = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If we can't create the first tensor, try with minimal tensor
            if (offset < Size) {
                x1 = torch::randn({1});
            } else {
                return 0;
            }
        }

        // Create second tensor
        torch::Tensor x2;
        try {
            x2 = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If we can't create the second tensor, create one matching x1's shape
            if (x1.dim() > 0) {
                x2 = torch::randn_like(x1);
            } else {
                x2 = torch::randn({1});
            }
        }

        // Parse dimension parameter
        int64_t dim = 1; // default
        if (offset < Size) {
            uint8_t dim_byte = Data[offset++];
            // Allow negative dimensions for testing
            dim = static_cast<int64_t>(static_cast<int8_t>(dim_byte));
            // Bound dimension to reasonable range
            if (dim > 10) dim = dim % 10;
            if (dim < -10) dim = -((-dim) % 10);
        }

        // Parse epsilon parameter
        double eps = 1e-8; // default
        if (offset < Size) {
            uint8_t eps_selector = Data[offset++];
            // Create various epsilon values including edge cases
            switch (eps_selector % 8) {
                case 0: eps = 0.0; break;
                case 1: eps = 1e-12; break;
                case 2: eps = 1e-8; break;
                case 3: eps = 1e-4; break;
                case 4: eps = 1.0; break;
                case 5: eps = -1e-8; break; // negative eps
                case 6: eps = std::numeric_limits<double>::min(); break;
                case 7: eps = std::numeric_limits<double>::denorm_min(); break;
            }
        }

        // Additional tensor manipulations based on remaining data
        if (offset < Size && (Data[offset++] % 4) == 0) {
            // Sometimes make tensors non-contiguous
            if (x1.dim() > 1 && x1.size(0) > 1 && x1.size(1) > 1) {
                x1 = x1.transpose(0, 1);
            }
        }
        
        if (offset < Size && (Data[offset++] % 4) == 0) {
            if (x2.dim() > 1 && x2.size(0) > 1 && x2.size(1) > 1) {
                x2 = x2.transpose(0, 1);
            }
        }

        // Sometimes add special values
        if (offset < Size) {
            uint8_t special_val = Data[offset++];
            if ((special_val % 8) == 0 && x1.numel() > 0) {
                x1.view(-1)[0] = std::numeric_limits<float>::infinity();
            } else if ((special_val % 8) == 1 && x1.numel() > 0) {
                x1.view(-1)[0] = -std::numeric_limits<float>::infinity();
            } else if ((special_val % 8) == 2 && x1.numel() > 0) {
                x1.view(-1)[0] = std::numeric_limits<float>::quiet_NaN();
            } else if ((special_val % 8) == 3 && x2.numel() > 0) {
                x2.view(-1)[0] = std::numeric_limits<float>::infinity();
            } else if ((special_val % 8) == 4 && x2.numel() > 0) {
                x2.view(-1)[0] = -std::numeric_limits<float>::infinity();
            } else if ((special_val % 8) == 5 && x2.numel() > 0) {
                x2.view(-1)[0] = std::numeric_limits<float>::quiet_NaN();
            } else if ((special_val % 8) == 6) {
                // Make all zeros in x1
                x1.zero_();
            } else if ((special_val % 8) == 7) {
                // Make all zeros in x2
                x2.zero_();
            }
        }

        // Test broadcasting scenarios
        if (offset < Size && (Data[offset++] % 3) == 0) {
            // Try to create broadcasting scenario
            if (x1.dim() > 0 && x2.dim() > 0) {
                auto shape1 = x1.sizes().vec();
                auto shape2 = x2.sizes().vec();
                
                // Make one dimension 1 for broadcasting
                if (shape1.size() > 0 && offset < Size) {
                    size_t idx = Data[offset++] % shape1.size();
                    shape1[idx] = 1;
                    try {
                        x1 = x1.reshape(shape1);
                    } catch (...) {
                        // Ignore reshape failures
                    }
                }
            }
        }

        // Sometimes requires grad
        if (offset < Size && (Data[offset++] % 2) == 0) {
            if (x1.dtype() == torch::kFloat || x1.dtype() == torch::kDouble || 
                x1.dtype() == torch::kHalf || x1.dtype() == torch::kBFloat16) {
                x1 = x1.requires_grad_(true);
            }
        }
        
        if (offset < Size && (Data[offset++] % 2) == 0) {
            if (x2.dtype() == torch::kFloat || x2.dtype() == torch::kDouble || 
                x2.dtype() == torch::kHalf || x2.dtype() == torch::kBFloat16) {
                x2 = x2.requires_grad_(true);
            }
        }

        // Main operation - cosine_similarity
        try {
            torch::Tensor result = torch::cosine_similarity(x1, x2, dim, eps);
            
            // Perform additional operations to increase coverage
            if (result.numel() > 0) {
                // Check for NaN/Inf
                bool has_nan = torch::isnan(result).any().item<bool>();
                bool has_inf = torch::isinf(result).any().item<bool>();
                
                // Sometimes compute backward if gradients are enabled
                if (result.requires_grad() && offset < Size && (Data[offset++] % 2) == 0) {
                    try {
                        auto sum_result = result.sum();
                        sum_result.backward();
                    } catch (...) {
                        // Ignore backward failures
                    }
                }
                
                // Test edge case: cosine similarity with itself
                if (offset < Size && (Data[offset++] % 3) == 0) {
                    try {
                        torch::Tensor self_sim = torch::cosine_similarity(x1, x1, dim, eps);
                    } catch (...) {
                        // Ignore failures
                    }
                }
            }
        } catch (const c10::Error& e) {
            // PyTorch errors are expected for invalid inputs
            // Continue fuzzing
        } catch (const std::runtime_error& e) {
            // Runtime errors from shape mismatches etc are expected
            // Continue fuzzing  
        }

        // Try with different dimensions
        if (offset < Size) {
            for (int d = -5; d <= 5 && offset < Size; ++d) {
                try {
                    torch::Tensor result2 = torch::cosine_similarity(x1, x2, d, eps);
                } catch (...) {
                    // Expected for invalid dimensions
                }
                offset++;
            }
        }

        // Test with explicit options
        if (x1.dim() > 0 && x2.dim() > 0) {
            try {
                // Try cosine_similarity with tensors on different devices if available
                if (torch::cuda::is_available() && offset < Size && (Data[offset++] % 4) == 0) {
                    auto x1_cuda = x1.to(torch::kCUDA);
                    auto x2_cuda = x2.to(torch::kCUDA);
                    torch::Tensor result_cuda = torch::cosine_similarity(x1_cuda, x2_cuda, dim, eps);
                }
            } catch (...) {
                // Ignore CUDA failures
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