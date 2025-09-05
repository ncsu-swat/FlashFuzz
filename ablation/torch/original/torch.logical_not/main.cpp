#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 3 bytes: 1 for operation mode, 2 for basic tensor creation
        if (Size < 3) {
            return 0;
        }
        
        // Parse operation mode
        uint8_t op_mode = Data[offset++];
        bool use_out_tensor = (op_mode & 0x01) != 0;
        bool out_same_dtype = (op_mode & 0x02) != 0;
        bool test_inplace = (op_mode & 0x04) != 0;
        bool test_empty_tensor = (op_mode & 0x08) != 0;
        bool test_scalar = (op_mode & 0x10) != 0;
        
        // Create input tensor
        torch::Tensor input_tensor;
        
        if (test_empty_tensor && offset < Size) {
            // Create an empty tensor with random shape
            uint8_t rank = Data[offset++] % 4 + 1;
            std::vector<int64_t> shape;
            bool has_zero = false;
            
            for (int i = 0; i < rank && offset < Size; i++) {
                int64_t dim = Data[offset++] % 5;
                if (dim == 0 && !has_zero) {
                    has_zero = true;  // Allow at most one zero dimension
                } else if (dim == 0) {
                    dim = 1;
                }
                shape.push_back(dim);
            }
            
            if (offset < Size) {
                auto dtype = fuzzer_utils::parseDataType(Data[offset++]);
                input_tensor = torch::empty(shape, torch::TensorOptions().dtype(dtype));
            } else {
                input_tensor = torch::empty(shape);
            }
        } else if (test_scalar && offset < Size) {
            // Create a scalar tensor
            auto dtype = fuzzer_utils::parseDataType(Data[offset++]);
            if (offset < Size) {
                // Parse a single value based on dtype
                if (dtype == torch::kBool) {
                    bool val = Data[offset++] & 0x01;
                    input_tensor = torch::tensor(val);
                } else if (dtype == torch::kFloat || dtype == torch::kDouble) {
                    float val = static_cast<float>(Data[offset++]) / 127.5f - 1.0f;
                    input_tensor = torch::tensor(val, torch::TensorOptions().dtype(dtype));
                } else {
                    int64_t val = static_cast<int64_t>(Data[offset++]) - 128;
                    input_tensor = torch::tensor(val, torch::TensorOptions().dtype(dtype));
                }
            } else {
                input_tensor = torch::zeros({}, torch::TensorOptions().dtype(dtype));
            }
        } else {
            // Create a regular tensor using fuzzer_utils
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Test basic logical_not operation
        torch::Tensor result = torch::logical_not(input_tensor);
        
        // Verify result has bool dtype unless specified otherwise
        if (result.dtype() != torch::kBool && !use_out_tensor) {
            std::cerr << "Warning: Result dtype is not bool: " << result.dtype() << std::endl;
        }
        
        // Test with output tensor if requested
        if (use_out_tensor && offset < Size) {
            torch::Tensor out_tensor;
            
            if (out_same_dtype || offset >= Size) {
                // Use same shape as input but potentially different dtype
                if (offset < Size) {
                    auto out_dtype = fuzzer_utils::parseDataType(Data[offset++]);
                    out_tensor = torch::empty(input_tensor.sizes(), 
                                            torch::TensorOptions().dtype(out_dtype));
                } else {
                    out_tensor = torch::empty(input_tensor.sizes());
                }
            } else {
                // Try to create a different shaped output tensor (should resize)
                try {
                    out_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                } catch (...) {
                    // Fall back to same shape
                    out_tensor = torch::empty(input_tensor.sizes());
                }
            }
            
            // Test logical_not with output tensor
            torch::logical_not(input_tensor, out_tensor);
            
            // Verify shapes match after operation
            if (out_tensor.sizes() != input_tensor.sizes()) {
                std::cerr << "Warning: Output tensor resized from " << out_tensor.sizes() 
                         << " to " << input_tensor.sizes() << std::endl;
            }
        }
        
        // Test in-place operation if tensor is bool type
        if (test_inplace && input_tensor.dtype() == torch::kBool) {
            torch::Tensor input_copy = input_tensor.clone();
            input_copy.logical_not_();
            
            // Verify in-place operation worked correctly
            if (!torch::equal(input_copy, result)) {
                std::cerr << "Warning: In-place logical_not produced different result" << std::endl;
            }
        }
        
        // Test edge cases with special values
        if (input_tensor.dtype() == torch::kFloat || input_tensor.dtype() == torch::kDouble) {
            // Test with NaN, Inf values if float type
            if (input_tensor.numel() > 0 && offset < Size) {
                uint8_t special_val = Data[offset++];
                if (special_val & 0x01) {
                    input_tensor.view(-1)[0] = std::numeric_limits<float>::quiet_NaN();
                }
                if ((special_val & 0x02) && input_tensor.numel() > 1) {
                    input_tensor.view(-1)[1] = std::numeric_limits<float>::infinity();
                }
                if ((special_val & 0x04) && input_tensor.numel() > 2) {
                    input_tensor.view(-1)[2] = -std::numeric_limits<float>::infinity();
                }
                
                // Re-test with special values
                torch::Tensor special_result = torch::logical_not(input_tensor);
                
                // NaN should be treated as True (non-zero), so logical_not(NaN) = False
                // Inf should be treated as True (non-zero), so logical_not(Inf) = False
            }
        }
        
        // Test with different memory layouts if enough data
        if (offset + 1 < Size && input_tensor.dim() >= 2) {
            uint8_t layout_type = Data[offset++];
            
            if (layout_type & 0x01) {
                // Test with transposed tensor
                torch::Tensor transposed = input_tensor.transpose(0, input_tensor.dim() - 1);
                torch::Tensor trans_result = torch::logical_not(transposed);
                
                // Verify consistency
                torch::Tensor expected = result.transpose(0, result.dim() - 1);
                if (!torch::equal(trans_result, expected)) {
                    std::cerr << "Warning: Transposed tensor gave inconsistent result" << std::endl;
                }
            }
            
            if (layout_type & 0x02) {
                // Test with non-contiguous tensor (sliced)
                if (input_tensor.size(0) > 1) {
                    torch::Tensor sliced = input_tensor.slice(0, 0, input_tensor.size(0), 2);
                    torch::Tensor slice_result = torch::logical_not(sliced);
                    
                    // Verify sliced result matches expected
                    torch::Tensor expected = result.slice(0, 0, result.size(0), 2);
                    if (!torch::equal(slice_result, expected)) {
                        std::cerr << "Warning: Sliced tensor gave inconsistent result" << std::endl;
                    }
                }
            }
        }
        
        // Test chained logical operations if enough data
        if (offset < Size) {
            uint8_t chain_op = Data[offset++];
            if (chain_op & 0x01) {
                // Double negation should give back original (as bool)
                torch::Tensor double_neg = torch::logical_not(torch::logical_not(input_tensor));
                torch::Tensor expected = input_tensor.to(torch::kBool);
                if (!torch::equal(double_neg, expected)) {
                    std::cerr << "Warning: Double negation doesn't match original" << std::endl;
                }
            }
        }
        
        // Test with requires_grad if floating point
        if ((input_tensor.dtype() == torch::kFloat || input_tensor.dtype() == torch::kDouble) 
            && offset < Size) {
            uint8_t grad_flag = Data[offset++];
            if (grad_flag & 0x01) {
                torch::Tensor grad_tensor = input_tensor.clone().requires_grad_(true);
                torch::Tensor grad_result = torch::logical_not(grad_tensor);
                
                // logical_not output should not require grad (it's discrete)
                if (grad_result.requires_grad()) {
                    std::cerr << "Warning: logical_not output unexpectedly requires grad" << std::endl;
                }
            }
        }
        
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors are expected during fuzzing
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        std::cout << "Unknown exception caught" << std::endl;
        return -1;
    }
    
    return 0;
}