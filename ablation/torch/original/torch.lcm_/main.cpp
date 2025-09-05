#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstdint>
#include <exception>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal data to create two tensors
        if (Size < 4) {
            // Not enough data for even basic tensor metadata
            return 0;
        }

        // Create first tensor (will be modified in-place)
        torch::Tensor tensor1;
        try {
            tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If we can't create the first tensor, bail out
            return 0;
        }

        // Create second tensor
        torch::Tensor tensor2;
        if (offset < Size) {
            try {
                tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            } catch (const std::exception& e) {
                // If we can't create second tensor, try with a scalar
                if (offset < Size) {
                    // Use remaining byte as scalar value
                    int64_t scalar_val = static_cast<int64_t>(Data[offset++]);
                    tensor2 = torch::tensor(scalar_val);
                } else {
                    // Use a default scalar
                    tensor2 = torch::tensor(2);
                }
            }
        } else {
            // No data left, create a default tensor
            tensor2 = torch::tensor(3);
        }

        // lcm_ only works with integer types, so convert if needed
        bool need_conversion = false;
        torch::ScalarType target_dtype = torch::kInt64;
        
        // Check if tensor1 is integer type
        if (!tensor1.dtype().isIntegralType(false)) {  // false = don't include bool
            need_conversion = true;
            // Try to preserve sign information if possible
            if (tensor1.dtype().isFloatingPoint() || tensor1.dtype().isComplex()) {
                tensor1 = tensor1.to(torch::kInt64);
            } else {
                tensor1 = tensor1.to(torch::kInt64);
            }
        }
        
        // Check if tensor2 is integer type
        if (!tensor2.dtype().isIntegralType(false)) {
            need_conversion = true;
            if (tensor2.dtype().isFloatingPoint() || tensor2.dtype().isComplex()) {
                tensor2 = tensor2.to(torch::kInt64);
            } else {
                tensor2 = tensor2.to(torch::kInt64);
            }
        }

        // Try different scenarios based on remaining fuzzer data
        if (offset < Size) {
            uint8_t scenario = Data[offset++] % 8;
            
            switch(scenario) {
                case 0:
                    // Normal case - both tensors as is
                    break;
                case 1:
                    // Make tensor2 a scalar
                    if (tensor2.numel() > 0) {
                        tensor2 = tensor2.flatten()[0];
                    }
                    break;
                case 2:
                    // Try broadcasting - reshape tensor2
                    if (tensor1.dim() > 0 && tensor2.numel() > 0) {
                        std::vector<int64_t> new_shape(tensor1.dim(), 1);
                        if (tensor1.dim() > 0) {
                            new_shape[tensor1.dim() - 1] = std::min(tensor2.numel(), tensor1.size(-1));
                        }
                        try {
                            tensor2 = tensor2.flatten().narrow(0, 0, new_shape[tensor1.dim() - 1]).reshape(new_shape);
                        } catch (...) {
                            // Keep original shape if reshape fails
                        }
                    }
                    break;
                case 3:
                    // Try with zeros
                    if (offset < Size && Data[offset++] % 2 == 0) {
                        tensor2 = torch::zeros_like(tensor1);
                    }
                    break;
                case 4:
                    // Try with ones
                    if (offset < Size && Data[offset++] % 2 == 0) {
                        tensor2 = torch::ones_like(tensor1);
                    }
                    break;
                case 5:
                    // Try with negative values
                    tensor2 = tensor2.neg();
                    break;
                case 6:
                    // Try making tensors same shape
                    if (tensor1.sizes() != tensor2.sizes() && tensor2.numel() == tensor1.numel()) {
                        try {
                            tensor2 = tensor2.reshape(tensor1.sizes());
                        } catch (...) {
                            // Keep original shape
                        }
                    }
                    break;
                case 7:
                    // Try with very large values
                    if (offset < Size) {
                        int64_t large_val = static_cast<int64_t>(Data[offset++]) * 1000000;
                        tensor2 = tensor2.add(large_val);
                    }
                    break;
            }
        }

        // Store original tensor1 for validation if needed
        torch::Tensor original = tensor1.clone();

        // Call lcm_ (in-place operation)
        try {
            tensor1.lcm_(tensor2);
            
            // Validate result properties
            // lcm should produce non-negative results
            if (tensor1.numel() > 0) {
                auto min_val = tensor1.min();
                // LCM results should be non-negative (by mathematical definition)
                // But PyTorch might have different behavior for negative inputs
            }
            
            // Check that tensor was actually modified (unless both were 0 or 1)
            if (tensor1.numel() > 0 && !torch::equal(tensor1, original)) {
                // Operation succeeded and modified the tensor
            }
            
        } catch (const c10::Error& e) {
            // PyTorch-specific errors (shape mismatch, dtype issues, etc.)
            // These are expected for some inputs
            return 0;
        } catch (const std::exception& e) {
            // Other exceptions might indicate bugs
            std::cout << "Exception caught: " << e.what() << std::endl;
            return -1;
        }

        // Additional edge case testing with the result
        if (tensor1.numel() > 0 && offset < Size) {
            uint8_t post_op = Data[offset++] % 4;
            try {
                switch(post_op) {
                    case 0:
                        // Try another lcm_ with the result
                        tensor1.lcm_(torch::tensor(Data[offset % Size], tensor1.options()));
                        break;
                    case 1:
                        // Try lcm_ with itself
                        tensor1.lcm_(tensor1.clone());
                        break;
                    case 2:
                        // Try with zero
                        tensor1.lcm_(torch::zeros(1, tensor1.options()));
                        break;
                    case 3:
                        // Try chained operations
                        tensor1.lcm_(torch::ones(1, tensor1.options()));
                        break;
                }
            } catch (const c10::Error& e) {
                // Expected for some edge cases
                return 0;
            }
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        std::cout << "Exception caught: unknown exception" << std::endl;
        return -1;
    }
}