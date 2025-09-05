#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        if (Size < 4) {
            // Need at least some bytes for basic parameters
            return 0;
        }

        size_t offset = 0;

        // Parse control parameters from fuzzer input
        uint8_t r_selector = Data[offset++];
        uint8_t replacement_flag = Data[offset++];
        uint8_t tensor_type_selector = Data[offset++];
        
        // Parse r (combination length) - keep it reasonable to avoid memory explosion
        int64_t r = (r_selector % 10) + 1; // Range [1, 10]
        
        // Parse with_replacement flag
        bool with_replacement = (replacement_flag & 1) == 1;
        
        // Decide tensor creation strategy based on selector
        torch::Tensor input_tensor;
        
        if (tensor_type_selector % 5 == 0 && offset < Size) {
            // Case 1: Empty tensor
            auto dtype = fuzzer_utils::parseDataType(Data[offset++]);
            input_tensor = torch::empty({0}, torch::TensorOptions().dtype(dtype));
        }
        else if (tensor_type_selector % 5 == 1 && offset < Size) {
            // Case 2: Single element tensor
            auto dtype = fuzzer_utils::parseDataType(Data[offset++]);
            input_tensor = torch::ones({1}, torch::TensorOptions().dtype(dtype));
        }
        else if (tensor_type_selector % 5 == 2) {
            // Case 3: Create tensor from fuzzer data
            try {
                input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                // Flatten to 1D as combinations expects 1D input
                input_tensor = input_tensor.flatten();
            } catch (const std::exception& e) {
                // If tensor creation fails, create a default tensor
                input_tensor = torch::arange(5, torch::kFloat32);
            }
        }
        else if (tensor_type_selector % 5 == 3 && offset + 2 < Size) {
            // Case 4: Specific size tensor with pattern
            uint8_t size_byte = Data[offset++];
            uint8_t dtype_byte = Data[offset++];
            int64_t tensor_size = (size_byte % 20) + 1; // Range [1, 20]
            auto dtype = fuzzer_utils::parseDataType(dtype_byte);
            
            // Create tensor with specific pattern based on remaining bytes
            if (offset < Size && Data[offset] % 3 == 0) {
                input_tensor = torch::arange(tensor_size, torch::TensorOptions().dtype(dtype));
            } else if (offset < Size && Data[offset] % 3 == 1) {
                input_tensor = torch::ones({tensor_size}, torch::TensorOptions().dtype(dtype));
            } else {
                input_tensor = torch::randn({tensor_size}, torch::TensorOptions().dtype(dtype));
            }
        }
        else {
            // Case 5: Default case - create tensor with various properties
            if (offset + 1 < Size) {
                uint8_t size_selector = Data[offset++];
                int64_t tensor_size = (size_selector % 15) + 1;
                
                // Try different tensor creation methods
                if (size_selector % 4 == 0) {
                    input_tensor = torch::linspace(0, 10, tensor_size);
                } else if (size_selector % 4 == 1) {
                    input_tensor = torch::randperm(tensor_size, torch::kInt64);
                } else if (size_selector % 4 == 2) {
                    // Tensor with repeated values
                    auto base = torch::arange(3);
                    input_tensor = base.repeat({(tensor_size / 3) + 1}).slice(0, 0, tensor_size);
                } else {
                    input_tensor = torch::rand({tensor_size});
                }
            } else {
                input_tensor = torch::arange(7);
            }
        }
        
        // Ensure tensor is 1D and has appropriate dtype for combinations
        if (input_tensor.dim() != 1) {
            input_tensor = input_tensor.flatten();
        }
        
        // Handle edge cases for r vs tensor size
        if (offset < Size) {
            uint8_t r_adjust = Data[offset++];
            if (r_adjust % 4 == 0) {
                // r equals tensor size
                r = input_tensor.size(0);
            } else if (r_adjust % 4 == 1) {
                // r greater than tensor size (interesting edge case)
                r = input_tensor.size(0) + (r_adjust % 5) + 1;
            } else if (r_adjust % 4 == 2) {
                // r is 0 (edge case)
                r = 0;
            }
            // else keep r as originally parsed
        }
        
        // Try to call combinations with various parameter combinations
        torch::Tensor result;
        
        try {
            // Main call to combinations
            result = torch::combinations(input_tensor, r, with_replacement);
            
            // Verify result properties
            if (result.defined()) {
                // Access result to ensure computation completes
                auto shape = result.sizes();
                auto dtype = result.dtype();
                
                // Perform some operations on result to increase coverage
                if (result.numel() > 0 && result.numel() < 10000) {
                    // Only for reasonable sizes to avoid memory issues
                    auto sum = result.sum();
                    if (result.dim() == 2 && result.size(0) > 0) {
                        auto first_row = result[0];
                        auto last_row = result[-1];
                    }
                }
                
                // Test edge case: combinations of combinations
                if (result.size(0) > 0 && result.size(0) <= 10 && offset < Size) {
                    uint8_t nested_flag = Data[offset++];
                    if (nested_flag % 10 == 0) {
                        auto flattened = result.flatten();
                        auto nested = torch::combinations(flattened, 2, false);
                    }
                }
            }
        } catch (const c10::Error& e) {
            // PyTorch-specific errors are expected for invalid inputs
            // Continue execution
        } catch (const std::exception& e) {
            // Other exceptions, log but continue
        }
        
        // Additional edge case testing with different tensor types
        if (offset + 2 < Size) {
            uint8_t extra_test = Data[offset++];
            if (extra_test % 5 == 0) {
                // Test with integer tensor
                auto int_tensor = torch::arange(5, torch::kInt32);
                auto int_result = torch::combinations(int_tensor, 2, false);
            } else if (extra_test % 5 == 1) {
                // Test with boolean tensor
                auto bool_tensor = torch::tensor({true, false, true, false});
                auto bool_result = torch::combinations(bool_tensor, 3, true);
            } else if (extra_test % 5 == 2) {
                // Test with negative values
                auto neg_tensor = torch::tensor({-1.0, -2.0, 3.0, -4.0});
                auto neg_result = torch::combinations(neg_tensor, 2, with_replacement);
            }
        }
        
        // Test boundary conditions
        if (offset < Size && Data[offset] % 3 == 0) {
            // Large r with small tensor
            auto small = torch::tensor({1.0, 2.0});
            try {
                auto res = torch::combinations(small, 10, true); // with_replacement allows this
            } catch (...) {}
            
            try {
                auto res = torch::combinations(small, 10, false); // should fail
            } catch (...) {}
        }
        
    }
    catch (const std::exception &e)
    {
        // Catch any uncaught exceptions to prevent fuzzer crashes
        // But still log them for debugging
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        // Catch any other exceptions
        std::cout << "Unknown exception caught" << std::endl;
        return -1;
    }
    
    return 0;
}