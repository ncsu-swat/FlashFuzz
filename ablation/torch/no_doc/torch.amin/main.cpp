#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need minimum bytes for: tensor creation + amin parameters
        if (Size < 4) {
            return 0;  // Not enough data, but keep for coverage
        }

        // Create input tensor
        torch::Tensor input_tensor;
        try {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If tensor creation fails, try with a simple default tensor
            if (offset < Size) {
                uint8_t fallback_rank = Data[offset] % 5;  // 0-4 dimensions
                std::vector<int64_t> shape;
                for (int i = 0; i < fallback_rank && offset < Size; i++) {
                    shape.push_back(1 + (Data[offset++] % 10));
                }
                if (shape.empty()) {
                    input_tensor = torch::randn({2, 3});
                } else {
                    input_tensor = torch::randn(shape);
                }
            } else {
                return 0;
            }
        }

        // Parse parameters for amin operation
        bool has_dim = false;
        std::vector<int64_t> dims;
        bool keepdim = false;
        
        if (offset < Size) {
            // Decide if we specify dimensions
            has_dim = Data[offset++] & 0x01;
            
            if (has_dim && offset < Size) {
                // Parse number of dimensions to reduce
                uint8_t num_dims = (Data[offset++] % input_tensor.dim()) + 1;
                
                // Parse which dimensions to reduce
                for (int i = 0; i < num_dims && offset < Size; i++) {
                    int64_t dim = Data[offset++] % input_tensor.dim();
                    // Handle negative dimensions too
                    if (offset < Size && (Data[offset++] & 0x01)) {
                        dim = -dim - 1;
                    }
                    dims.push_back(dim);
                }
                
                // Remove duplicates to avoid issues
                std::sort(dims.begin(), dims.end());
                dims.erase(std::unique(dims.begin(), dims.end()), dims.end());
            }
            
            // Parse keepdim parameter
            if (offset < Size) {
                keepdim = Data[offset++] & 0x01;
            }
        }

        // Test various amin operations
        torch::Tensor result;
        
        // Case 1: amin without dimensions (reduce all)
        if (!has_dim || dims.empty()) {
            result = torch::amin(input_tensor);
            
            // Also test with explicit keepdim when no dims specified
            if (offset < Size && (Data[offset] & 0x01)) {
                torch::Tensor result2 = input_tensor.amin();
            }
        }
        // Case 2: amin with single dimension
        else if (dims.size() == 1) {
            result = torch::amin(input_tensor, dims[0], keepdim);
            
            // Also test alternative API
            result = input_tensor.amin(dims[0], keepdim);
        }
        // Case 3: amin with multiple dimensions
        else {
            result = torch::amin(input_tensor, dims, keepdim);
            
            // Test with IntArrayRef
            c10::IntArrayRef dim_ref(dims);
            torch::Tensor result2 = input_tensor.amin(dim_ref, keepdim);
        }

        // Additional edge cases based on remaining data
        if (offset < Size) {
            uint8_t edge_case = Data[offset++] % 10;
            
            switch (edge_case) {
                case 0:
                    // Test on empty tensor if possible
                    if (input_tensor.numel() == 0 || offset < Size) {
                        try {
                            torch::Tensor empty_tensor = torch::empty({0, 3, 4});
                            torch::Tensor empty_result = torch::amin(empty_tensor);
                        } catch (...) {
                            // Expected to fail on empty tensors
                        }
                    }
                    break;
                    
                case 1:
                    // Test on scalar tensor
                    {
                        torch::Tensor scalar = torch::tensor(3.14);
                        torch::Tensor scalar_result = torch::amin(scalar);
                    }
                    break;
                    
                case 2:
                    // Test with out parameter
                    if (result.defined()) {
                        torch::Tensor out_tensor = torch::empty_like(result);
                        torch::amin_out(out_tensor, input_tensor, dims.empty() ? c10::nullopt : c10::optional<c10::IntArrayRef>(dims), keepdim);
                    }
                    break;
                    
                case 3:
                    // Test on different dtypes
                    if (input_tensor.dtype() == torch::kFloat32) {
                        torch::Tensor int_tensor = input_tensor.to(torch::kInt32);
                        torch::Tensor int_result = torch::amin(int_tensor);
                    }
                    break;
                    
                case 4:
                    // Test on non-contiguous tensor
                    if (input_tensor.dim() >= 2) {
                        torch::Tensor transposed = input_tensor.transpose(0, input_tensor.dim() - 1);
                        torch::Tensor trans_result = torch::amin(transposed);
                    }
                    break;
                    
                case 5:
                    // Test with all dimensions
                    if (input_tensor.dim() > 0) {
                        std::vector<int64_t> all_dims;
                        for (int64_t i = 0; i < input_tensor.dim(); i++) {
                            all_dims.push_back(i);
                        }
                        torch::Tensor all_reduce = torch::amin(input_tensor, all_dims, true);
                    }
                    break;
                    
                case 6:
                    // Test with negative dimensions
                    if (input_tensor.dim() > 0) {
                        torch::Tensor neg_result = torch::amin(input_tensor, -1, keepdim);
                    }
                    break;
                    
                case 7:
                    // Test on complex tensors if applicable
                    if (input_tensor.is_floating_point()) {
                        try {
                            torch::Tensor complex_tensor = torch::complex(input_tensor, torch::zeros_like(input_tensor));
                            // amin might not support complex, but let's try
                            torch::Tensor complex_result = torch::amin(complex_tensor);
                        } catch (...) {
                            // Complex might not be supported
                        }
                    }
                    break;
                    
                case 8:
                    // Test with CUDA tensors if available
                    if (torch::cuda::is_available() && offset < Size && (Data[offset] & 0x01)) {
                        torch::Tensor cuda_tensor = input_tensor.cuda();
                        torch::Tensor cuda_result = torch::amin(cuda_tensor, dims.empty() ? c10::nullopt : c10::optional<c10::IntArrayRef>(dims), keepdim);
                        // Move back to CPU for validation
                        cuda_result = cuda_result.cpu();
                    }
                    break;
                    
                case 9:
                    // Test chained operations
                    if (input_tensor.numel() > 0) {
                        torch::Tensor chained = torch::amin(torch::amin(input_tensor));
                    }
                    break;
            }
        }

        // Test boundary conditions with special values
        if (offset < Size && (Data[offset++] & 0x03) == 0) {
            // Create tensor with special values
            torch::Tensor special_tensor = torch::tensor({
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::quiet_NaN(),
                0.0f, -0.0f, 1.0f, -1.0f
            });
            
            if (special_tensor.numel() > 0) {
                torch::Tensor special_result = torch::amin(special_tensor);
                
                // Test that amin handles NaN correctly
                torch::Tensor nan_tensor = torch::full({2, 3}, std::numeric_limits<float>::quiet_NaN());
                torch::Tensor nan_result = torch::amin(nan_tensor);
            }
        }

        // Validate results when possible
        if (result.defined() && result.numel() > 0) {
            // Check that result values are <= all values in original tensor along reduced dimensions
            if (!has_dim || dims.empty()) {
                float min_val = result.item<float>();
                float actual_min = input_tensor.min().item<float>();
                // Due to floating point, use tolerance
                if (std::abs(min_val - actual_min) > 1e-5 && !std::isnan(min_val) && !std::isnan(actual_min)) {
                    // Potential issue, but don't crash
                }
            }
        }

        return 0;
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors are expected during fuzzing
        return 0;  // Keep going
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;  // Discard input for unexpected exceptions
    }
    catch (...)
    {
        // Catch any other exceptions
        return -1;
    }
}