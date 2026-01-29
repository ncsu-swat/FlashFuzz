#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <vector>

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
        
        // Determine number of tensors to use (2-5)
        if (Size < 4) return 0;
        uint8_t num_tensors = (Data[offset++] % 4) + 2; // 2 to 5 tensors
        
        // Determine dtype (must be floating point for matmul)
        uint8_t dtype_selector = Data[offset++] % 3;
        torch::ScalarType dtype;
        switch (dtype_selector) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Create a vector to store our tensors
        std::vector<torch::Tensor> tensors;
        
        // Generate dimensions for chain multiplication
        // For chain A1*A2*...*An, we need dimensions: (d0,d1), (d1,d2), ..., (d_{n-1},d_n)
        std::vector<int64_t> dims;
        for (uint8_t i = 0; i <= num_tensors && offset < Size; ++i) {
            int64_t dim = (Data[offset++] % 8) + 1; // 1-8
            dims.push_back(dim);
        }
        
        // Ensure we have enough dimensions
        while (dims.size() <= static_cast<size_t>(num_tensors)) {
            dims.push_back(2); // default dimension
        }
        
        // Create tensors with compatible dimensions
        for (uint8_t i = 0; i < num_tensors; ++i) {
            int64_t rows = dims[i];
            int64_t cols = dims[i + 1];
            
            // Create tensor with the determined shape
            torch::Tensor tensor;
            if (offset + 1 < Size) {
                uint8_t init_type = Data[offset++] % 4;
                switch (init_type) {
                    case 0:
                        tensor = torch::randn({rows, cols}, torch::TensorOptions().dtype(dtype));
                        break;
                    case 1:
                        tensor = torch::ones({rows, cols}, torch::TensorOptions().dtype(dtype));
                        break;
                    case 2:
                        tensor = torch::zeros({rows, cols}, torch::TensorOptions().dtype(dtype));
                        break;
                    default:
                        tensor = torch::rand({rows, cols}, torch::TensorOptions().dtype(dtype));
                        break;
                }
            } else {
                tensor = torch::randn({rows, cols}, torch::TensorOptions().dtype(dtype));
            }
            
            tensors.push_back(tensor);
        }
        
        // Ensure we have at least 2 tensors
        if (tensors.size() < 2) {
            tensors.clear();
            tensors.push_back(torch::randn({2, 3}, torch::TensorOptions().dtype(dtype)));
            tensors.push_back(torch::randn({3, 2}, torch::TensorOptions().dtype(dtype)));
        }
        
        // Apply chain_matmul operation
        torch::Tensor result;
        try {
            result = torch::chain_matmul(tensors);
        } catch (const c10::Error& e) {
            // PyTorch specific errors (shape mismatches, etc.) are expected
            return 0;
        }
        
        // Verify result shape
        if (result.defined() && result.numel() > 0) {
            auto first_tensor = tensors.front();
            auto last_tensor = tensors.back();
            
            int64_t expected_rows = first_tensor.size(0);
            int64_t expected_cols = last_tensor.size(1);
            
            // Validate result dimensions
            if (result.dim() != 2 || 
                result.size(0) != expected_rows || 
                result.size(1) != expected_cols) {
                std::cerr << "Unexpected result shape: expected [" << expected_rows 
                          << ", " << expected_cols << "], got " << result.sizes() << std::endl;
            }
            
            // Exercise the result tensor to ensure it's valid
            auto sum = result.sum();
            auto mean = result.mean();
            
            // Additional operations to increase coverage
            if (offset < Size && (Data[offset] % 2 == 0)) {
                auto transposed = result.t();
                auto contiguous = result.contiguous();
            }
        }
        
        // Test with TensorList overload (same as vector but different call pattern)
        if (tensors.size() >= 2 && offset < Size && (Data[offset] % 3 == 0)) {
            try {
                torch::TensorList tensor_list(tensors);
                auto result2 = torch::chain_matmul(tensor_list);
            } catch (const c10::Error& e) {
                // Expected for some inputs
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}