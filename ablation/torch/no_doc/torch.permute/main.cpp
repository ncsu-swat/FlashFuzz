#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>

// Helper to generate permutation from fuzzer data
std::vector<int64_t> generatePermutation(const uint8_t* data, size_t& offset, size_t size, int64_t rank) {
    std::vector<int64_t> perm;
    
    if (rank == 0) {
        // Scalar tensor - no permutation needed
        return perm;
    }
    
    // Initialize with identity permutation
    perm.resize(rank);
    std::iota(perm.begin(), perm.end(), 0);
    
    // Use fuzzer data to shuffle the permutation
    if (offset < size) {
        uint8_t shuffle_type = data[offset++];
        
        switch (shuffle_type % 5) {
            case 0:
                // Identity permutation (already set)
                break;
            case 1:
                // Reverse permutation
                std::reverse(perm.begin(), perm.end());
                break;
            case 2:
                // Swap first and last
                if (rank > 1) {
                    std::swap(perm[0], perm[rank - 1]);
                }
                break;
            case 3:
                // Random shuffle based on fuzzer data
                for (int64_t i = 0; i < rank && offset < size; ++i) {
                    int64_t j = data[offset++] % rank;
                    std::swap(perm[i], perm[j]);
                }
                break;
            case 4:
                // Potentially invalid permutation (duplicate indices or out of range)
                if (offset + rank <= size) {
                    for (int64_t i = 0; i < rank; ++i) {
                        // Allow invalid values to test error handling
                        perm[i] = static_cast<int64_t>(data[offset++]) - 128;
                    }
                }
                break;
        }
    }
    
    return perm;
}

// Helper to test various permute scenarios
void testPermuteVariations(torch::Tensor& tensor, const uint8_t* data, size_t& offset, size_t size) {
    // Test 1: Standard permute with vector
    try {
        auto perm = generatePermutation(data, offset, size, tensor.dim());
        auto result = tensor.permute(perm);
        
        // Verify shape transformation
        if (result.defined()) {
            // Access some properties to ensure tensor is valid
            auto shape = result.sizes();
            auto stride = result.strides();
            auto numel = result.numel();
            
            // Test chained operations
            if (offset < size && data[offset++] % 2 == 0) {
                auto result2 = result.permute(perm);  // Double permute
            }
        }
    } catch (const c10::Error& e) {
        // Expected for invalid permutations
    } catch (const std::exception& e) {
        // Other exceptions
    }
    
    // Test 2: Permute with initializer list (if rank is small)
    if (tensor.dim() == 2 && offset < size) {
        try {
            auto result = tensor.permute({1, 0});  // Transpose for 2D
        } catch (const c10::Error& e) {
            // Expected for some edge cases
        }
    } else if (tensor.dim() == 3 && offset < size) {
        try {
            auto result = tensor.permute({2, 0, 1});  // Rotate dimensions
        } catch (const c10::Error& e) {
            // Expected for some edge cases
        }
    }
    
    // Test 3: Edge cases with negative indices
    if (tensor.dim() > 0 && offset < size && data[offset++] % 3 == 0) {
        try {
            std::vector<int64_t> neg_perm;
            for (int64_t i = 0; i < tensor.dim(); ++i) {
                neg_perm.push_back(-tensor.dim() + i);  // Use negative indexing
            }
            auto result = tensor.permute(neg_perm);
        } catch (const c10::Error& e) {
            // Expected for invalid negative indices
        }
    }
    
    // Test 4: Empty permutation vector (should fail)
    if (offset < size && data[offset++] % 4 == 0) {
        try {
            std::vector<int64_t> empty_perm;
            auto result = tensor.permute(empty_perm);
        } catch (const c10::Error& e) {
            // Expected to fail
        }
    }
    
    // Test 5: Permutation with too many/few dimensions
    if (offset < size) {
        uint8_t size_modifier = data[offset++];
        try {
            std::vector<int64_t> wrong_size_perm;
            int64_t wrong_size = (size_modifier % 3 == 0) ? tensor.dim() + 1 : 
                                 (size_modifier % 3 == 1) ? tensor.dim() - 1 : 
                                 size_modifier % 10;
            for (int64_t i = 0; i < wrong_size; ++i) {
                wrong_size_perm.push_back(i % std::max(int64_t(1), tensor.dim()));
            }
            auto result = tensor.permute(wrong_size_perm);
        } catch (const c10::Error& e) {
            // Expected for dimension mismatch
        }
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        // Minimum size check
        if (size < 3) {
            return 0;  // Need at least some bytes for tensor creation
        }
        
        // Create primary tensor
        torch::Tensor tensor;
        try {
            tensor = fuzzer_utils::createTensor(data, size, offset);
        } catch (const std::exception& e) {
            // If we can't create a tensor, try with a default one
            tensor = torch::randn({2, 3});
        }
        
        // Test various permute operations
        testPermuteVariations(tensor, data, offset, size);
        
        // Additional tests for special tensor types
        if (offset < size) {
            uint8_t special_type = data[offset++];
            
            switch (special_type % 6) {
                case 0: {
                    // Test with empty tensor
                    auto empty_tensor = torch::empty({0, 3, 2});
                    testPermuteVariations(empty_tensor, data, offset, size);
                    break;
                }
                case 1: {
                    // Test with scalar tensor
                    auto scalar_tensor = torch::tensor(3.14);
                    testPermuteVariations(scalar_tensor, data, offset, size);
                    break;
                }
                case 2: {
                    // Test with 1D tensor
                    auto tensor_1d = torch::randn({5});
                    testPermuteVariations(tensor_1d, data, offset, size);
                    break;
                }
                case 3: {
                    // Test with high-dimensional tensor
                    auto tensor_nd = torch::randn({2, 1, 3, 1, 2});
                    testPermuteVariations(tensor_nd, data, offset, size);
                    break;
                }
                case 4: {
                    // Test with non-contiguous tensor
                    auto base = torch::randn({4, 5, 6});
                    auto non_contig = base.transpose(0, 2);
                    testPermuteVariations(non_contig, data, offset, size);
                    break;
                }
                case 5: {
                    // Test with view of a tensor
                    auto base = torch::randn({12});
                    auto view = base.view({3, 4});
                    testPermuteVariations(view, data, offset, size);
                    break;
                }
            }
        }
        
        // Test permute with requires_grad
        if (offset < size && data[offset++] % 2 == 0) {
            try {
                auto grad_tensor = torch::randn({3, 4, 5}, torch::requires_grad());
                auto perm = generatePermutation(data, offset, size, grad_tensor.dim());
                auto result = grad_tensor.permute(perm);
                
                // Test backward pass
                if (result.defined() && result.requires_grad()) {
                    auto loss = result.sum();
                    loss.backward();
                }
            } catch (const c10::Error& e) {
                // Expected for some invalid operations
            }
        }
        
        // Test permute on different devices (if available)
        if (torch::cuda::is_available() && offset < size && data[offset++] % 10 == 0) {
            try {
                auto cuda_tensor = tensor.to(torch::kCUDA);
                testPermuteVariations(cuda_tensor, data, offset, size);
            } catch (const c10::Error& e) {
                // CUDA operations might fail
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected for edge cases
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;  // Discard input for unexpected exceptions
    } catch (...) {
        // Catch any other exceptions
        std::cout << "Exception caught: Unknown exception" << std::endl;
        return -1;
    }
    
    return 0;  // Keep the input
}