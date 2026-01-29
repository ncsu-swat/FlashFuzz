#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <algorithm>      // For std::max

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
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Read control bytes for tensor configuration
        uint8_t num_tensors = (Data[offset++] % 4) + 1; // 1-4 tensors
        uint8_t base_dims = (Data[offset++] % 3) + 1;   // 1-3 dimensions
        
        // Create tensors with compatible dimensions for nesting
        std::vector<torch::Tensor> tensor_list;
        
        for (int i = 0; i < num_tensors && offset + 4 < Size; i++) {
            // Create shape with same number of dimensions but potentially different sizes
            std::vector<int64_t> shape;
            for (int d = 0; d < base_dims; d++) {
                int64_t dim_size = (Data[offset++] % 8) + 1; // 1-8 per dimension
                if (offset >= Size) break;
                shape.push_back(dim_size);
            }
            
            if (shape.empty()) {
                shape.push_back(1);
            }
            
            // Create tensor with the shape
            torch::Tensor t = torch::randn(shape);
            tensor_list.push_back(t);
        }
        
        if (tensor_list.empty()) {
            // Need at least one tensor
            tensor_list.push_back(torch::randn({2, 3}));
        }
        
        // 1. Create a nested tensor from a list of tensors
        torch::Tensor nested_tensor = torch::nested::nested_tensor(tensor_list);
        
        // 2. Test nested tensor properties
        bool is_nested = nested_tensor.is_nested();
        int64_t ndim = nested_tensor.dim();
        
        // 3. Test to_padded_tensor conversion
        if (offset < Size) {
            double padding_value = static_cast<double>(Data[offset++]) / 255.0 - 0.5;
            try {
                auto padded = torch::nested::to_padded_tensor(nested_tensor, padding_value);
                // Verify padding worked
                auto padded_sizes = padded.sizes();
            } catch (const std::exception&) {
                // Some nested tensor configurations may not support padding
            }
        }
        
        // 4. Test unbind operation (unbind at dim 0 returns constituent tensors)
        try {
            auto unbind_result = nested_tensor.unbind(0);
            // Verify we got the right number of tensors back
            if (unbind_result.size() != tensor_list.size()) {
                // Unexpected behavior
            }
        } catch (const std::exception&) {
            // May fail for certain configurations
        }
        
        // 5. Test with different dtypes
        if (offset < Size) {
            uint8_t dtype_choice = Data[offset++] % 4;
            torch::ScalarType dtype;
            switch (dtype_choice) {
                case 0: dtype = torch::kFloat32; break;
                case 1: dtype = torch::kFloat64; break;
                case 2: dtype = torch::kFloat16; break;
                default: dtype = torch::kFloat32; break;
            }
            
            try {
                std::vector<torch::Tensor> typed_list;
                for (const auto& t : tensor_list) {
                    typed_list.push_back(t.to(dtype));
                }
                auto nested_typed = torch::nested::nested_tensor(typed_list);
                
                // Test operations on typed nested tensor
                auto padded_typed = torch::nested::to_padded_tensor(nested_typed, 0.0);
            } catch (const std::exception&) {
                // Some dtypes may not be supported for nested tensors
            }
        }
        
        // 6. Test nested tensor with requires_grad
        if (offset < Size && (Data[offset++] & 0x01)) {
            try {
                std::vector<torch::Tensor> grad_list;
                for (const auto& t : tensor_list) {
                    grad_list.push_back(t.clone().requires_grad_(true));
                }
                auto nested_grad = torch::nested::nested_tensor(grad_list);
                
                // Test that gradients flow through
                auto padded_grad = torch::nested::to_padded_tensor(nested_grad, 0.0);
                auto sum_val = padded_grad.sum();
                sum_val.backward();
            } catch (const std::exception&) {
                // Gradient operations may fail in some cases
            }
        }
        
        // 7. Test as_nested_tensor if available (convert padded back to nested)
        if (offset < Size) {
            try {
                auto padded = torch::nested::to_padded_tensor(nested_tensor, 0.0);
                // padded is a regular tensor now, test some operations
                auto padded_sum = padded.sum();
                auto padded_mean = padded.mean();
            } catch (const std::exception&) {
                // May fail for certain configurations
            }
        }
        
        // 8. Test nested tensor indexing
        if (tensor_list.size() > 0) {
            try {
                // Access first element using select
                auto first = nested_tensor.select(0, 0);
            } catch (const std::exception&) {
                // Indexing may not be fully supported
            }
        }
        
        // 9. Test clone and contiguous
        try {
            auto cloned = nested_tensor.clone();
            // Note: contiguous() may behave differently for nested tensors
        } catch (const std::exception&) {
            // May fail
        }
        
        // 10. Test device operations (CPU)
        try {
            auto on_cpu = nested_tensor.to(torch::kCPU);
        } catch (const std::exception&) {
            // May fail
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}