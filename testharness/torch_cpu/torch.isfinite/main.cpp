#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply isfinite operation
        torch::Tensor result = torch::isfinite(input_tensor);
        
        // Access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            try {
                bool has_true = result.any().item<bool>();
                bool has_false = torch::logical_not(result).any().item<bool>();
                (void)has_true;
                (void)has_false;
                
                torch::Tensor sum_result = result.sum();
                torch::Tensor mean_result = result.to(torch::kFloat).mean();
                (void)sum_result;
                (void)mean_result;
            } catch (...) {
                // Silently ignore errors from result operations
            }
        }
        
        // Test with floating point tensors containing special values
        if (offset + 2 < Size) {
            // Create tensors with special values directly
            std::vector<torch::Dtype> float_dtypes = {
                torch::kFloat32, torch::kFloat64
            };
            
            uint8_t dtype_idx = Data[offset % Size] % float_dtypes.size();
            torch::Dtype dtype = float_dtypes[dtype_idx];
            
            // Create tensor with inf values
            torch::Tensor inf_tensor = torch::full({3, 3}, std::numeric_limits<double>::infinity(), 
                                                    torch::TensorOptions().dtype(dtype));
            torch::Tensor inf_result = torch::isfinite(inf_tensor);
            
            // Create tensor with negative inf values
            torch::Tensor neg_inf_tensor = torch::full({2, 4}, -std::numeric_limits<double>::infinity(),
                                                        torch::TensorOptions().dtype(dtype));
            torch::Tensor neg_inf_result = torch::isfinite(neg_inf_tensor);
            
            // Create tensor with NaN values
            torch::Tensor nan_tensor = torch::full({4, 2}, std::nan(""),
                                                    torch::TensorOptions().dtype(dtype));
            torch::Tensor nan_result = torch::isfinite(nan_tensor);
            
            // Create mixed tensor using cat
            torch::Tensor finite_vals = torch::tensor({1.0, 2.0, 3.0}, 
                                                       torch::TensorOptions().dtype(dtype));
            torch::Tensor inf_vals = torch::tensor({std::numeric_limits<double>::infinity()},
                                                    torch::TensorOptions().dtype(dtype));
            torch::Tensor nan_vals = torch::tensor({std::nan("")},
                                                    torch::TensorOptions().dtype(dtype));
            torch::Tensor mixed = torch::cat({finite_vals, inf_vals, nan_vals});
            torch::Tensor mixed_result = torch::isfinite(mixed);
            
            // Verify result dtype is bool
            if (mixed_result.dtype() != torch::kBool) {
                std::cerr << "Unexpected result dtype" << std::endl;
            }
            
            // Test with zero-dimensional tensor (scalar)
            torch::Tensor scalar = torch::tensor(42.0, torch::TensorOptions().dtype(dtype));
            torch::Tensor scalar_result = torch::isfinite(scalar);
            
            // Test with empty tensor
            torch::Tensor empty = torch::empty({0}, torch::TensorOptions().dtype(dtype));
            torch::Tensor empty_result = torch::isfinite(empty);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}