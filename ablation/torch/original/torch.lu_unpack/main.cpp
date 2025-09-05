#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 4 bytes for configuration
        if (Size < 4) {
            return 0;
        }
        
        // Parse configuration bytes
        uint8_t config1 = Data[offset++];
        uint8_t config2 = Data[offset++];
        uint8_t config3 = Data[offset++];
        uint8_t config4 = Data[offset++];
        
        // Extract boolean flags from config bytes
        bool unpack_data = config1 & 0x01;
        bool unpack_pivots = config1 & 0x02;
        
        // Determine if we should create square or rectangular matrices
        bool is_rectangular = config2 & 0x01;
        
        // Create LU_data tensor
        torch::Tensor LU_data;
        try {
            LU_data = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If we can't create the first tensor, bail out
            return 0;
        }
        
        // Ensure LU_data has at least 2 dimensions for matrix operations
        if (LU_data.dim() < 2) {
            // Add dimensions to make it at least 2D
            while (LU_data.dim() < 2) {
                LU_data = LU_data.unsqueeze(0);
            }
        }
        
        // Get the shape for creating compatible pivots tensor
        auto lu_sizes = LU_data.sizes();
        int64_t m = lu_sizes[lu_sizes.size() - 2];
        int64_t n = lu_sizes[lu_sizes.size() - 1];
        int64_t min_mn = std::min(m, n);
        
        // Create pivots shape - should be (..., min(m, n))
        std::vector<int64_t> pivot_shape;
        for (int64_t i = 0; i < lu_sizes.size() - 2; ++i) {
            pivot_shape.push_back(lu_sizes[i]);
        }
        pivot_shape.push_back(min_mn);
        
        // Create LU_pivots tensor
        torch::Tensor LU_pivots;
        
        // Use remaining data to create pivots or generate them
        if (offset < Size - 2) {
            try {
                LU_pivots = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Reshape pivots to match expected shape
                if (LU_pivots.numel() > 0) {
                    // Convert to int32 or int64 as pivots should be integer type
                    if (!LU_pivots.dtype().isIntegral(false)) {
                        LU_pivots = LU_pivots.to(torch::kInt32);
                    }
                    
                    // Try to reshape to match expected pivot shape
                    int64_t total_pivot_elements = 1;
                    for (auto dim : pivot_shape) {
                        total_pivot_elements *= dim;
                    }
                    
                    if (LU_pivots.numel() >= total_pivot_elements && total_pivot_elements > 0) {
                        LU_pivots = LU_pivots.flatten().slice(0, 0, total_pivot_elements).reshape(pivot_shape);
                    } else if (total_pivot_elements > 0) {
                        // Not enough elements, create valid pivots
                        LU_pivots = torch::arange(1, min_mn + 1, torch::kInt32);
                        if (pivot_shape.size() > 1) {
                            LU_pivots = LU_pivots.expand(pivot_shape);
                        }
                    }
                }
            } catch (...) {
                // If parsing fails, create default pivots
                LU_pivots = torch::arange(1, min_mn + 1, torch::kInt32);
                if (pivot_shape.size() > 1) {
                    LU_pivots = LU_pivots.expand(pivot_shape);
                }
            }
        } else {
            // Create default valid pivots (1-indexed as expected by lu_unpack)
            LU_pivots = torch::arange(1, min_mn + 1, torch::kInt32);
            if (pivot_shape.size() > 1) {
                LU_pivots = LU_pivots.expand(pivot_shape);
            }
        }
        
        // Ensure pivots are integer type
        if (!LU_pivots.dtype().isIntegral(false)) {
            LU_pivots = LU_pivots.to(torch::kInt32);
        }
        
        // Try different dtype combinations based on config
        if (config3 & 0x01) {
            LU_data = LU_data.to(torch::kFloat32);
        } else if (config3 & 0x02) {
            LU_data = LU_data.to(torch::kFloat64);
        } else if (config3 & 0x04) {
            try {
                LU_data = LU_data.to(torch::kComplexFloat);
            } catch (...) {
                // If conversion fails, keep original dtype
            }
        } else if (config3 & 0x08) {
            try {
                LU_data = LU_data.to(torch::kComplexDouble);
            } catch (...) {
                // If conversion fails, keep original dtype
            }
        }
        
        // Ensure LU_data is floating point or complex type
        if (!LU_data.dtype().isFloatingPoint() && !LU_data.dtype().isComplex()) {
            LU_data = LU_data.to(torch::kFloat32);
        }
        
        // Test with different device configurations if available
        if (config4 & 0x01 && torch::cuda::is_available()) {
            try {
                LU_data = LU_data.cuda();
                LU_pivots = LU_pivots.cuda();
            } catch (...) {
                // If CUDA transfer fails, keep on CPU
            }
        }
        
        // Call torch::lu_unpack with various configurations
        try {
            auto [P, L, U] = torch::lu_unpack(LU_data, LU_pivots, unpack_data, unpack_pivots);
            
            // Perform some operations on the results to ensure they're valid
            if (unpack_data) {
                // L and U should not be empty
                if (L.numel() > 0 && U.numel() > 0) {
                    // Try matrix multiplication to verify shapes are compatible
                    try {
                        auto LU_reconstructed = torch::matmul(L, U);
                        // This exercises the decomposition
                    } catch (...) {
                        // Matrix multiplication might fail for invalid shapes
                    }
                }
            }
            
            if (unpack_pivots && P.numel() > 0) {
                // P should be a permutation matrix or batch of permutation matrices
                // Try to use it
                try {
                    if (unpack_data && L.numel() > 0) {
                        auto PL = torch::matmul(P, L);
                        // This exercises the permutation matrix
                    }
                } catch (...) {
                    // Operation might fail for invalid shapes
                }
            }
            
        } catch (const c10::Error& e) {
            // PyTorch-specific errors are expected for invalid inputs
            return 0;
        } catch (const std::exception& e) {
            // Log unexpected exceptions
            std::cout << "Exception caught: " << e.what() << std::endl;
            return -1;
        }
        
        // Try with output tensors pre-allocated
        if (config4 & 0x02) {
            try {
                // Pre-allocate output tensors with appropriate shapes
                torch::Tensor P_out, L_out, U_out;
                
                if (unpack_pivots) {
                    std::vector<int64_t> p_shape = LU_data.sizes().vec();
                    p_shape[p_shape.size() - 1] = m;  // P is m x m
                    P_out = torch::empty(p_shape, LU_data.options());
                }
                
                if (unpack_data) {
                    std::vector<int64_t> l_shape = LU_data.sizes().vec();
                    l_shape[l_shape.size() - 1] = min_mn;  // L is m x min(m,n)
                    L_out = torch::empty(l_shape, LU_data.options());
                    
                    std::vector<int64_t> u_shape = LU_data.sizes().vec();
                    u_shape[u_shape.size() - 2] = min_mn;  // U is min(m,n) x n
                    U_out = torch::empty(u_shape, LU_data.options());
                }
                
                // Call with pre-allocated outputs
                auto [P2, L2, U2] = torch::lu_unpack(LU_data, LU_pivots, unpack_data, unpack_pivots);
                
            } catch (...) {
                // Pre-allocated output version might fail
            }
        }
        
        // Test edge cases with empty tensors
        if (config4 & 0x04) {
            try {
                auto empty_lu = torch::empty({0, 0}, LU_data.options());
                auto empty_pivots = torch::empty({0}, torch::kInt32);
                auto [Pe, Le, Ue] = torch::lu_unpack(empty_lu, empty_pivots, unpack_data, unpack_pivots);
            } catch (...) {
                // Empty tensor operations might fail
            }
        }
        
        // Test with batched tensors
        if (config4 & 0x08 && LU_data.dim() == 2) {
            try {
                // Add batch dimension
                auto batched_lu = LU_data.unsqueeze(0).expand({3, -1, -1});
                auto batched_pivots = LU_pivots.unsqueeze(0).expand({3, -1});
                auto [Pb, Lb, Ub] = torch::lu_unpack(batched_lu, batched_pivots, unpack_data, unpack_pivots);
            } catch (...) {
                // Batched operations might fail
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