#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create mask tensor (must be boolean type)
        torch::Tensor mask;
        if (offset < Size) {
            mask = fuzzer_utils::createTensor(Data, Size, offset);
            // Convert to boolean tensor regardless of original type
            mask = mask.to(torch::kBool);
        } else {
            // If we don't have enough data, create a mask of the same shape as input
            mask = torch::ones_like(input, torch::kBool);
        }
        
        // Try different broadcasting scenarios
        if (offset < Size && Data[offset] % 3 == 0) {
            // Case 1: Same shape
            // Already handled by default
        } else if (offset < Size && Data[offset] % 3 == 1) {
            // Case 2: Mask is a scalar
            try {
                if (mask.numel() > 0) {
                    mask = mask.flatten()[0].reshape({});
                } else {
                    mask = torch::tensor(true, torch::kBool);
                }
            } catch (...) {
                // Keep original mask on failure
            }
        } else {
            // Case 3: Try to create a mask with different but broadcastable shape
            if (input.dim() > 0 && offset + 1 < Size) {
                std::vector<int64_t> new_shape;
                for (int i = 0; i < input.dim(); i++) {
                    if (i < input.dim() - 1 || (offset < Size && Data[offset] % 2 == 0)) {
                        new_shape.push_back(1);
                    } else {
                        new_shape.push_back(input.size(i));
                    }
                }
                
                if (!new_shape.empty()) {
                    try {
                        mask = mask.reshape(new_shape);
                    } catch (...) {
                        // If reshape fails, keep original mask
                    }
                }
            }
        }
        
        if (offset < Size) {
            offset++;
        }
        
        // Apply masked_select operation
        torch::Tensor result = torch::masked_select(input, mask);
        
        // Try some edge cases
        if (offset < Size) {
            uint8_t edge_case = Data[offset++];
            
            try {
                if (edge_case % 5 == 0) {
                    // Empty tensor
                    torch::Tensor empty_input = torch::empty({0}, input.options());
                    torch::Tensor empty_mask = torch::empty({0}, torch::kBool);
                    result = torch::masked_select(empty_input, empty_mask);
                } else if (edge_case % 5 == 1) {
                    // All false mask
                    torch::Tensor all_false = torch::zeros_like(input, torch::kBool);
                    result = torch::masked_select(input, all_false);
                } else if (edge_case % 5 == 2) {
                    // All true mask
                    torch::Tensor all_true = torch::ones_like(input, torch::kBool);
                    result = torch::masked_select(input, all_true);
                } else if (edge_case % 5 == 3) {
                    // Scalar input with scalar mask
                    if (input.numel() > 0) {
                        torch::Tensor scalar_input = input.flatten()[0].reshape({});
                        torch::Tensor scalar_mask = torch::tensor(true, torch::kBool);
                        result = torch::masked_select(scalar_input, scalar_mask);
                    }
                } else {
                    // Non-contiguous tensors
                    if (input.dim() >= 2 && input.size(0) > 1 && input.size(1) > 1) {
                        torch::Tensor non_contig_input = input.transpose(0, 1);
                        torch::Tensor non_contig_mask = mask;
                        if (mask.dim() >= 2 && mask.size(0) > 1 && mask.size(1) > 1) {
                            non_contig_mask = mask.transpose(0, 1);
                        }
                        result = torch::masked_select(non_contig_input, non_contig_mask);
                    }
                }
            } catch (...) {
                // Edge cases may fail due to shape mismatches, etc. - silently continue
            }
        }
        
        // Access result to ensure computation is not optimized away
        if (result.numel() > 0) {
            try {
                // Use sum() which works for all dtypes
                volatile auto sum_val = result.sum().item<double>();
                (void)sum_val;
            } catch (...) {
                // Ignore access errors
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}