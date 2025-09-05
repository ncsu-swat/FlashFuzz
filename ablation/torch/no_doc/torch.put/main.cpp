#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 3 bytes for control flags and minimal tensor creation
        if (Size < 3) {
            return 0;
        }

        // Parse control flags
        uint8_t accumulate_flag = Data[offset++];
        bool accumulate = accumulate_flag & 0x01;
        
        // Create input tensor
        torch::Tensor input_tensor;
        try {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If we can't create the first tensor, bail out
            return 0;
        }
        
        // Create index tensor
        torch::Tensor index_tensor;
        try {
            index_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            // Convert to long type as indices must be integer type
            if (!index_tensor.dtype().isIntegral(false)) {
                index_tensor = index_tensor.to(torch::kLong);
            } else if (index_tensor.scalar_type() != torch::kLong) {
                index_tensor = index_tensor.to(torch::kLong);
            }
        } catch (const std::exception& e) {
            // Try with a default index tensor
            index_tensor = torch::zeros({1}, torch::kLong);
        }
        
        // Create source tensor
        torch::Tensor source_tensor;
        try {
            source_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            // Convert source to match input dtype if different
            if (source_tensor.dtype() != input_tensor.dtype()) {
                source_tensor = source_tensor.to(input_tensor.dtype());
            }
        } catch (const std::exception& e) {
            // Try with a default source tensor matching input dtype
            source_tensor = torch::ones({1}, input_tensor.options());
        }
        
        // Flatten tensors as put operates on flattened views
        torch::Tensor input_flat = input_tensor.flatten();
        torch::Tensor index_flat = index_tensor.flatten();
        torch::Tensor source_flat = source_tensor.flatten();
        
        // Ensure indices are within valid range for the flattened input
        int64_t input_numel = input_flat.numel();
        if (input_numel > 0) {
            // Clamp indices to valid range [0, input_numel-1] or handle negative indices
            index_flat = index_flat.remainder(input_numel);
            index_flat = index_flat.abs();
        }
        
        // Handle size mismatches between index and source
        int64_t index_numel = index_flat.numel();
        int64_t source_numel = source_flat.numel();
        
        if (index_numel > 0 && source_numel > 0) {
            // Adjust sizes if needed
            if (index_numel > source_numel) {
                // Truncate index to match source size
                index_flat = index_flat.slice(0, 0, source_numel);
            } else if (source_numel > index_numel) {
                // Truncate source to match index size
                source_flat = source_flat.slice(0, 0, index_numel);
            }
        }
        
        // Try the put operation
        try {
            torch::Tensor result = input_flat.put(index_flat, source_flat, accumulate);
            
            // Reshape result back to original shape
            result = result.reshape(input_tensor.sizes());
            
            // Exercise the result tensor to ensure computation completes
            if (result.numel() > 0) {
                auto sum = result.sum();
                // Force computation
                sum.item<float>();
            }
            
        } catch (const c10::Error& e) {
            // PyTorch-specific errors are expected for invalid operations
            return 0;
        } catch (const std::runtime_error& e) {
            // Runtime errors from invalid tensor operations
            return 0;
        }
        
        // Also test in-place version
        try {
            torch::Tensor input_copy = input_flat.clone();
            input_copy.put_(index_flat, source_flat, accumulate);
            
            // Verify in-place operation completed
            if (input_copy.numel() > 0) {
                auto max_val = input_copy.max();
                max_val.item<float>();
            }
        } catch (const c10::Error& e) {
            // Expected for some invalid configurations
            return 0;
        } catch (const std::runtime_error& e) {
            return 0;
        }
        
        // Test edge cases with different tensor configurations
        if (offset < Size) {
            uint8_t edge_case = Data[offset++];
            
            // Test with empty tensors
            if (edge_case & 0x01) {
                try {
                    torch::Tensor empty_input = torch::empty({0});
                    torch::Tensor empty_index = torch::empty({0}, torch::kLong);
                    torch::Tensor empty_source = torch::empty({0});
                    auto result = empty_input.put(empty_index, empty_source, false);
                } catch (...) {
                    // Ignore errors for edge cases
                }
            }
            
            // Test with scalar tensors
            if (edge_case & 0x02) {
                try {
                    torch::Tensor scalar_input = torch::ones({});
                    torch::Tensor scalar_index = torch::zeros({}, torch::kLong);
                    torch::Tensor scalar_source = torch::ones({}) * 2;
                    auto result = scalar_input.put(scalar_index, scalar_source, accumulate);
                } catch (...) {
                    // Ignore errors for edge cases
                }
            }
            
            // Test with out-of-bounds indices (should be handled by remainder operation above)
            if (edge_case & 0x04 && input_numel > 0) {
                try {
                    torch::Tensor oob_index = torch::tensor({input_numel * 2, -input_numel * 2}, torch::kLong);
                    torch::Tensor oob_source = torch::ones({2}, input_tensor.options());
                    auto result = input_flat.clone().put(oob_index, oob_source, false);
                } catch (...) {
                    // Expected to potentially fail
                }
            }
            
            // Test with duplicate indices
            if (edge_case & 0x08 && input_numel > 0) {
                try {
                    torch::Tensor dup_index = torch::zeros({3}, torch::kLong);
                    torch::Tensor dup_source = torch::arange(3, input_tensor.options());
                    auto result = input_flat.clone().put(dup_index, dup_source, accumulate);
                } catch (...) {
                    // Ignore errors
                }
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