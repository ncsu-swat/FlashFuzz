#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal bytes for creating tensors
        if (Size < 4) {
            return 0;
        }

        // Create the main tensor for LU decomposition
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a 2D tensor for LU decomposition
        if (input_tensor.dim() < 2) {
            // Reshape to 2D if needed
            int64_t total_elements = input_tensor.numel();
            if (total_elements == 0) {
                return 0;
            }
            // Create a square-ish matrix
            int64_t dim = std::max(int64_t(1), static_cast<int64_t>(std::sqrt(total_elements)));
            int64_t rows = dim;
            int64_t cols = total_elements / dim;
            if (cols == 0) cols = 1;
            input_tensor = input_tensor.reshape({rows, cols});
        }
        
        // LU decomposition requires floating point types
        if (!input_tensor.is_floating_point() && !input_tensor.is_complex()) {
            // Convert to float
            input_tensor = input_tensor.to(torch::kFloat32);
        }
        
        // Make the tensor square if it's not (LU typically works on square matrices)
        if (input_tensor.size(0) != input_tensor.size(1)) {
            int64_t min_dim = std::min(input_tensor.size(0), input_tensor.size(1));
            if (min_dim > 0) {
                input_tensor = input_tensor.narrow(0, 0, min_dim).narrow(1, 0, min_dim);
            }
        }
        
        // Perform LU decomposition first to get LU_data and LU_pivots
        torch::Tensor LU_data, LU_pivots;
        
        try {
            // Try standard LU decomposition
            auto lu_result = torch::lu(input_tensor);
            LU_data = std::get<0>(lu_result);
            LU_pivots = std::get<1>(lu_result);
        } catch (...) {
            // If standard LU fails, create synthetic LU_data and pivots from fuzzer data
            if (offset < Size) {
                // Create LU_data tensor (should be same shape as input)
                LU_data = input_tensor.clone();
                
                // Create pivots tensor (should be 1D with length = min(m, n))
                int64_t pivot_size = input_tensor.size(0);
                if (offset + pivot_size <= Size) {
                    std::vector<int32_t> pivot_data;
                    for (int64_t i = 0; i < pivot_size; ++i) {
                        if (offset < Size) {
                            // Generate pivot indices (1-indexed for LU)
                            int32_t pivot = (Data[offset++] % pivot_size) + 1;
                            pivot_data.push_back(pivot);
                        } else {
                            pivot_data.push_back(i + 1);
                        }
                    }
                    LU_pivots = torch::from_blob(pivot_data.data(), {pivot_size}, 
                                                torch::kInt32).clone();
                } else {
                    // Default pivots
                    LU_pivots = torch::arange(1, pivot_size + 1, torch::kInt32);
                }
            } else {
                // Fallback: use identity-like decomposition
                LU_data = input_tensor.clone();
                LU_pivots = torch::arange(1, input_tensor.size(0) + 1, torch::kInt32);
            }
        }
        
        // Test lu_unpack with various configurations
        
        // 1. Standard lu_unpack
        try {
            auto [P1, L1, U1] = torch::lu_unpack(LU_data, LU_pivots);
            
            // Verify shapes
            if (P1.dim() != 2 || L1.dim() != 2 || U1.dim() != 2) {
                std::cerr << "Unexpected output dimensions" << std::endl;
            }
            
            // Test reconstruction: P @ L @ U should approximate original
            auto reconstructed = torch::matmul(torch::matmul(P1, L1), U1);
        } catch (const std::exception& e) {
            // Continue testing other scenarios
        }
        
        // 2. Test with different pivot ranges if we have more data
        if (offset < Size && Size - offset >= 2) {
            int64_t m = LU_data.size(0);
            int64_t n = LU_data.size(1);
            
            // Parse dimensions for unpack
            bool unpack_data = (Data[offset++] % 2) == 0;
            bool unpack_pivots = (Data[offset++] % 2) == 0;
            
            try {
                auto [P2, L2, U2] = torch::lu_unpack(LU_data, LU_pivots, 
                                                     unpack_data, unpack_pivots);
            } catch (...) {
                // Expected for some invalid combinations
            }
        }
        
        // 3. Test with non-contiguous tensors
        if (LU_data.numel() > 1) {
            try {
                auto LU_data_transposed = LU_data.transpose(0, 1);
                auto [P3, L3, U3] = torch::lu_unpack(LU_data_transposed.transpose(0, 1), 
                                                     LU_pivots);
            } catch (...) {
                // Continue
            }
        }
        
        // 4. Test with different data types
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            torch::ScalarType new_dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            if (new_dtype == torch::kFloat64 || new_dtype == torch::kComplexFloat || 
                new_dtype == torch::kComplexDouble) {
                try {
                    auto LU_data_cast = LU_data.to(new_dtype);
                    auto [P4, L4, U4] = torch::lu_unpack(LU_data_cast, LU_pivots);
                } catch (...) {
                    // Continue
                }
            }
        }
        
        // 5. Test with batch dimensions
        if (offset < Size && LU_data.numel() > 0) {
            uint8_t batch_size = (Data[offset++] % 4) + 1;
            try {
                // Create batched version
                auto LU_data_batched = LU_data.unsqueeze(0).repeat({batch_size, 1, 1});
                auto LU_pivots_batched = LU_pivots.unsqueeze(0).repeat({batch_size, 1});
                
                auto [P5, L5, U5] = torch::lu_unpack(LU_data_batched, LU_pivots_batched);
                
                // Verify batch dimensions
                if (P5.size(0) != batch_size || L5.size(0) != batch_size || 
                    U5.size(0) != batch_size) {
                    std::cerr << "Batch dimension mismatch" << std::endl;
                }
            } catch (...) {
                // Continue
            }
        }
        
        // 6. Edge case: empty tensors
        try {
            auto empty_data = torch::empty({0, 0}, LU_data.options());
            auto empty_pivots = torch::empty({0}, torch::kInt32);
            auto [P6, L6, U6] = torch::lu_unpack(empty_data, empty_pivots);
        } catch (...) {
            // Expected for empty tensors
        }
        
        // 7. Test with manually crafted invalid pivots
        if (LU_pivots.numel() > 0) {
            try {
                // Create pivots with out-of-range values
                auto bad_pivots = torch::full_like(LU_pivots, 999);
                auto [P7, L7, U7] = torch::lu_unpack(LU_data, bad_pivots);
            } catch (...) {
                // Expected for invalid pivots
            }
            
            try {
                // Create pivots with negative values
                auto neg_pivots = torch::full_like(LU_pivots, -1);
                auto [P8, L8, U8] = torch::lu_unpack(LU_data, neg_pivots);
            } catch (...) {
                // Expected for invalid pivots
            }
        }
        
        // 8. Test with mismatched dimensions
        if (LU_pivots.numel() > 1) {
            try {
                // Wrong pivot size
                auto wrong_pivots = LU_pivots.narrow(0, 0, LU_pivots.size(0) / 2);
                auto [P9, L9, U9] = torch::lu_unpack(LU_data, wrong_pivots);
            } catch (...) {
                // Expected for dimension mismatch
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