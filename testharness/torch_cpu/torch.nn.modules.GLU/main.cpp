#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

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
        
        // Need at least a few bytes to create a tensor and parameters
        if (Size < 8) {
            return 0;
        }
        
        // Extract dim parameter first (before tensor creation)
        int8_t dim_byte = static_cast<int8_t>(Data[offset++]);
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get tensor dimensions
        int64_t ndim = input.dim();
        if (ndim == 0) {
            return 0;  // GLU doesn't work on 0-dim tensors
        }
        
        // Normalize dim to valid range
        int64_t dim = dim_byte % ndim;
        if (dim < 0) {
            dim += ndim;
        }
        
        // GLU requires the dimension size to be even
        // Ensure the tensor has even size along the specified dimension
        int64_t dim_size = input.size(dim);
        if (dim_size < 2 || dim_size % 2 != 0) {
            // Reshape or pad to make it even
            // Create a new tensor with appropriate size
            std::vector<int64_t> sizes = input.sizes().vec();
            sizes[dim] = std::max(int64_t(2), ((dim_size / 2) + 1) * 2);
            input = torch::randn(sizes);
        }
        
        // Create GLU module and apply
        try {
            torch::nn::GLU glu_module(torch::nn::GLUOptions().dim(dim));
            torch::Tensor output = glu_module->forward(input);
        } catch (const std::exception&) {
            // Shape mismatch or invalid dim - expected
        }
        
        // Try functional version
        try {
            torch::Tensor functional_output = torch::nn::functional::glu(input, 
                torch::nn::functional::GLUFuncOptions().dim(dim));
        } catch (const std::exception&) {
            // Expected for invalid inputs
        }
        
        // Try with different dimensions if there's more data
        if (offset < Size) {
            int8_t dim_byte2 = static_cast<int8_t>(Data[offset++]);
            int64_t dim2 = dim_byte2 % ndim;
            if (dim2 < 0) {
                dim2 += ndim;
            }
            
            // Ensure even size along dim2
            int64_t dim2_size = input.size(dim2);
            torch::Tensor input2 = input;
            if (dim2_size >= 2 && dim2_size % 2 == 0) {
                try {
                    torch::nn::GLU glu_module2(torch::nn::GLUOptions().dim(dim2));
                    torch::Tensor output2 = glu_module2->forward(input2);
                } catch (const std::exception&) {
                    // Expected
                }
            }
        }
        
        // Try with negative dimension (valid in PyTorch)
        try {
            int64_t neg_dim = -(dim + 1);  // Convert to negative indexing
            torch::nn::GLU glu_module_neg(torch::nn::GLUOptions().dim(neg_dim));
            torch::Tensor output_neg = glu_module_neg->forward(input);
        } catch (const std::exception&) {
            // Expected for invalid inputs
        }
        
        // Test with different tensor types
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++] % 3;
            try {
                torch::Tensor typed_input;
                switch (dtype_selector) {
                    case 0:
                        typed_input = input.to(torch::kFloat32);
                        break;
                    case 1:
                        typed_input = input.to(torch::kFloat64);
                        break;
                    case 2:
                        typed_input = input.to(torch::kFloat16);
                        break;
                }
                torch::nn::GLU glu_typed(torch::nn::GLUOptions().dim(dim));
                torch::Tensor output_typed = glu_typed->forward(typed_input);
            } catch (const std::exception&) {
                // Some dtypes may not be supported
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}