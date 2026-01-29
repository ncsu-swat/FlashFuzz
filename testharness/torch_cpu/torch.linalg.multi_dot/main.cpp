#include "fuzzer_utils.h"
#include <iostream>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 4) return 0;
        
        size_t offset = 0;
        
        // Parse the number of tensors (between 2 and 5)
        uint8_t num_tensors = (Data[offset++] % 4) + 2;
        
        // Parse dimensions for the chain
        // For multi_dot: tensor[i] has shape (d[i], d[i+1])
        // We need num_tensors + 1 dimension values
        std::vector<int64_t> dims;
        for (int i = 0; i <= num_tensors; ++i) {
            if (offset >= Size) {
                dims.push_back(2); // Default dimension
            } else {
                // Limit dimensions to reasonable range (1-16) to avoid memory issues
                dims.push_back((Data[offset++] % 16) + 1);
            }
        }
        
        // Create tensors with compatible dimensions
        std::vector<torch::Tensor> tensors;
        
        // Determine dtype from fuzzer input
        torch::ScalarType dtype = torch::kFloat;
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++] % 4;
            switch (dtype_selector) {
                case 0: dtype = torch::kFloat; break;
                case 1: dtype = torch::kDouble; break;
                case 2: dtype = torch::kComplexFloat; break;
                case 3: dtype = torch::kComplexDouble; break;
            }
        }
        
        for (int i = 0; i < num_tensors; ++i) {
            int64_t rows = dims[i];
            int64_t cols = dims[i + 1];
            
            // Create tensor with proper shape for matrix multiplication chain
            torch::Tensor tensor = torch::randn({rows, cols}, torch::TensorOptions().dtype(dtype));
            tensors.push_back(tensor);
        }
        
        // Apply torch::linalg_multi_dot (C++ API uses underscore naming)
        torch::Tensor result = torch::linalg_multi_dot(tensors);
        
        // Verify result shape: should be (dims[0], dims[num_tensors])
        if (result.size(0) != dims[0] || result.size(1) != dims[num_tensors]) {
            std::cerr << "Unexpected result shape" << std::endl;
        }
        
        // Test with 1D tensors at the ends (special case)
        if (offset < Size && Data[offset++] % 2 == 0 && num_tensors >= 2) {
            try {
                std::vector<torch::Tensor> tensors_1d;
                
                // First tensor can be 1D (vector)
                tensors_1d.push_back(torch::randn({dims[0]}, torch::TensorOptions().dtype(dtype)));
                
                // Middle tensors are 2D
                for (int i = 1; i < num_tensors - 1; ++i) {
                    tensors_1d.push_back(torch::randn({dims[i], dims[i + 1]}, torch::TensorOptions().dtype(dtype)));
                }
                
                // Last tensor can be 1D (vector)
                if (num_tensors > 1) {
                    tensors_1d.push_back(torch::randn({dims[num_tensors - 1]}, torch::TensorOptions().dtype(dtype)));
                }
                
                if (tensors_1d.size() >= 2) {
                    torch::Tensor result_1d = torch::linalg_multi_dot(tensors_1d);
                }
            } catch (const std::exception &) {
                // 1D edge cases may fail, that's expected
            }
        }
        
        // Test with contiguous vs non-contiguous tensors
        if (offset < Size && Data[offset++] % 3 == 0) {
            try {
                std::vector<torch::Tensor> nc_tensors;
                for (const auto& t : tensors) {
                    // Make non-contiguous by transposing twice with different memory layout
                    if (t.dim() == 2) {
                        nc_tensors.push_back(t.t().t().clone());
                    } else {
                        nc_tensors.push_back(t.clone());
                    }
                }
                torch::Tensor nc_result = torch::linalg_multi_dot(nc_tensors);
            } catch (const std::exception &) {
                // May fail for some configurations
            }
        }
        
        // Test with single-element dimensions
        if (offset < Size && Data[offset++] % 4 == 0) {
            try {
                std::vector<torch::Tensor> small_tensors;
                small_tensors.push_back(torch::randn({1, 3}, torch::TensorOptions().dtype(dtype)));
                small_tensors.push_back(torch::randn({3, 1}, torch::TensorOptions().dtype(dtype)));
                torch::Tensor small_result = torch::linalg_multi_dot(small_tensors);
            } catch (const std::exception &) {
                // Expected for edge cases
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