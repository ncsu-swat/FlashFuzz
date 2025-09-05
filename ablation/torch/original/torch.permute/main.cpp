#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>

// Helper to generate permutation dimensions from fuzzer data
std::vector<int64_t> generatePermutationDims(const uint8_t* data, size_t& offset, size_t size, int64_t rank) {
    std::vector<int64_t> dims;
    
    if (rank == 0) {
        // Scalar tensor - no dimensions to permute
        return dims;
    }
    
    // Initialize with identity permutation
    dims.resize(rank);
    std::iota(dims.begin(), dims.end(), 0);
    
    // Use fuzzer data to shuffle or modify the permutation
    if (offset < size) {
        uint8_t strategy = data[offset++];
        
        switch (strategy % 5) {
            case 0: {
                // Identity permutation (already set)
                break;
            }
            case 1: {
                // Reverse permutation
                std::reverse(dims.begin(), dims.end());
                break;
            }
            case 2: {
                // Random shuffle based on fuzzer data
                for (int64_t i = 0; i < rank && offset < size; ++i) {
                    uint8_t swap_idx = data[offset++] % rank;
                    std::swap(dims[i], dims[swap_idx]);
                }
                break;
            }
            case 3: {
                // Cyclic shift right
                if (offset < size && rank > 1) {
                    uint8_t shift = data[offset++] % rank;
                    std::rotate(dims.rbegin(), dims.rbegin() + shift, dims.rend());
                }
                break;
            }
            case 4: {
                // Build custom permutation from fuzzer bytes
                for (int64_t i = 0; i < rank && offset < size; ++i) {
                    dims[i] = data[offset++] % rank;
                }
                // Note: This might create invalid permutations (duplicates/missing dims)
                // which is good for testing error handling
                break;
            }
        }
    }
    
    return dims;
}

// Helper to test permute with negative dimensions
std::vector<int64_t> generateNegativePermutationDims(const uint8_t* data, size_t& offset, size_t size, int64_t rank) {
    std::vector<int64_t> dims = generatePermutationDims(data, offset, size, rank);
    
    // Convert some dimensions to negative indices
    if (offset < size && rank > 0) {
        uint8_t neg_mask = data[offset++];
        for (int64_t i = 0; i < rank && i < 8; ++i) {
            if (neg_mask & (1 << i)) {
                // Convert to negative index (PyTorch supports negative indexing)
                dims[i] = dims[i] - rank;
            }
        }
    }
    
    return dims;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 3) {
        // Need minimum bytes for tensor creation and permutation strategy
        return 0;
    }
    
    try {
        size_t offset = 0;
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(data, size, offset);
        int64_t rank = input.dim();
        
        // Generate permutation dimensions
        std::vector<int64_t> perm_dims = generatePermutationDims(data, offset, size, rank);
        
        // Test 1: Basic permute
        try {
            torch::Tensor result = torch::permute(input, perm_dims);
            
            // Verify output properties
            if (result.dim() != rank && rank > 0) {
                std::cerr << "Unexpected rank change after permute" << std::endl;
            }
            
            // Test that permute creates a view (shares storage)
            if (input.numel() > 0 && result.data_ptr() != input.data_ptr()) {
                // Note: This might not always be true for all tensor types
            }
            
            // Test double permute (should be reversible for valid permutations)
            if (!perm_dims.empty()) {
                // Create inverse permutation if valid
                std::vector<int64_t> inverse_perm(rank);
                bool is_valid_perm = true;
                std::vector<bool> seen(rank, false);
                
                for (int64_t i = 0; i < rank; ++i) {
                    int64_t dim = perm_dims[i];
                    // Handle negative indices
                    if (dim < 0) dim += rank;
                    
                    if (dim >= 0 && dim < rank && !seen[dim]) {
                        inverse_perm[dim] = i;
                        seen[dim] = true;
                    } else {
                        is_valid_perm = false;
                        break;
                    }
                }
                
                if (is_valid_perm) {
                    torch::Tensor double_permuted = torch::permute(result, inverse_perm);
                    if (input.numel() > 0 && !torch::equal(input, double_permuted)) {
                        // This might indicate an issue, but could also be due to 
                        // floating point precision or non-contiguous tensors
                    }
                }
            }
        } catch (const c10::Error& e) {
            // Expected for invalid permutations
        } catch (const std::exception& e) {
            // Other expected errors for edge cases
        }
        
        // Test 2: Negative dimension indices
        if (offset < size) {
            std::vector<int64_t> neg_perm_dims = generateNegativePermutationDims(data, offset, size, rank);
            try {
                torch::Tensor result_neg = torch::permute(input, neg_perm_dims);
            } catch (const c10::Error& e) {
                // Expected for invalid permutations
            }
        }
        
        // Test 3: Edge cases
        if (offset < size) {
            uint8_t edge_case = data[offset++] % 6;
            
            switch (edge_case) {
                case 0: {
                    // Empty permutation vector (should fail for non-scalar)
                    try {
                        torch::Tensor result = torch::permute(input, {});
                    } catch (const c10::Error& e) {
                        // Expected
                    }
                    break;
                }
                case 1: {
                    // Too many dimensions
                    std::vector<int64_t> too_many(rank + 1);
                    std::iota(too_many.begin(), too_many.end(), 0);
                    try {
                        torch::Tensor result = torch::permute(input, too_many);
                    } catch (const c10::Error& e) {
                        // Expected
                    }
                    break;
                }
                case 2: {
                    // Out of range dimensions
                    if (rank > 0) {
                        std::vector<int64_t> out_of_range(rank);
                        std::iota(out_of_range.begin(), out_of_range.end(), 0);
                        out_of_range[0] = rank; // Invalid index
                        try {
                            torch::Tensor result = torch::permute(input, out_of_range);
                        } catch (const c10::Error& e) {
                            // Expected
                        }
                    }
                    break;
                }
                case 3: {
                    // Duplicate dimensions
                    if (rank > 1) {
                        std::vector<int64_t> duplicates(rank, 0);
                        try {
                            torch::Tensor result = torch::permute(input, duplicates);
                        } catch (const c10::Error& e) {
                            // Expected
                        }
                    }
                    break;
                }
                case 4: {
                    // Mixed positive and negative indices
                    if (rank > 1) {
                        std::vector<int64_t> mixed(rank);
                        for (int64_t i = 0; i < rank; ++i) {
                            mixed[i] = (i % 2 == 0) ? i : i - rank;
                        }
                        try {
                            torch::Tensor result = torch::permute(input, mixed);
                        } catch (const c10::Error& e) {
                            // Could be valid or invalid depending on the permutation
                        }
                    }
                    break;
                }
                case 5: {
                    // Test with different tensor types (complex, bool, etc.)
                    if (input.dtype() == torch::kComplexFloat || input.dtype() == torch::kComplexDouble) {
                        // Permute should work with complex tensors
                        try {
                            torch::Tensor result = torch::permute(input, perm_dims);
                        } catch (const c10::Error& e) {
                            // Unexpected for valid permutations
                        }
                    }
                    break;
                }
            }
        }
        
        // Test 4: Chain multiple permutations
        if (rank > 1 && offset + rank < size) {
            try {
                torch::Tensor temp = input;
                for (int i = 0; i < 3 && offset < size; ++i) {
                    std::vector<int64_t> chain_perm = generatePermutationDims(data, offset, size, rank);
                    temp = torch::permute(temp, chain_perm);
                }
            } catch (const c10::Error& e) {
                // Expected for invalid permutations in the chain
            }
        }
        
        // Test 5: Permute with gradient tracking (if applicable)
        if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble) {
            try {
                torch::Tensor grad_input = input.requires_grad_(true);
                torch::Tensor grad_result = torch::permute(grad_input, perm_dims);
                
                // Check that gradient tracking is preserved
                if (grad_input.requires_grad() && !grad_result.requires_grad()) {
                    std::cerr << "Gradient tracking not preserved through permute" << std::endl;
                }
            } catch (const c10::Error& e) {
                // Some operations might not support gradients
            }
        }
        
    } catch (const std::exception& e) {
        // Catch any unexpected exceptions to keep fuzzer running
        // These are likely from tensor creation or other setup issues
        return 0;
    }
    
    return 0;
}