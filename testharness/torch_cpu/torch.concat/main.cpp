#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Need at least 4 bytes for meaningful input
        if (Size < 4) {
            return 0;
        }
        
        // Determine number of tensors to concatenate (2-5)
        uint8_t num_tensors = (Data[offset++] % 4) + 2;
        
        // Determine dimension to concatenate along (0-3)
        int64_t cat_dim = Data[offset++] % 4;
        
        // Determine tensor rank (1-4)
        uint8_t rank = (Data[offset++] % 4) + 1;
        
        // Ensure cat_dim is valid for the rank
        if (cat_dim >= rank) {
            cat_dim = rank - 1;
        }
        
        // Determine base shape from fuzzer data
        std::vector<int64_t> base_shape;
        for (uint8_t i = 0; i < rank && offset < Size; ++i) {
            int64_t dim_size = (Data[offset++] % 8) + 1;  // 1-8 for each dimension
            base_shape.push_back(dim_size);
        }
        
        // Pad shape if we didn't get enough bytes
        while (base_shape.size() < rank) {
            base_shape.push_back(2);
        }
        
        // Determine dtype
        torch::ScalarType dtype = torch::kFloat32;
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++] % 4;
            switch (dtype_selector) {
                case 0: dtype = torch::kFloat32; break;
                case 1: dtype = torch::kFloat64; break;
                case 2: dtype = torch::kInt32; break;
                case 3: dtype = torch::kInt64; break;
            }
        }
        
        // Create tensors with compatible shapes
        std::vector<torch::Tensor> tensors;
        for (uint8_t i = 0; i < num_tensors; ++i) {
            std::vector<int64_t> tensor_shape = base_shape;
            
            // Vary only the concatenation dimension
            if (offset < Size) {
                tensor_shape[cat_dim] = (Data[offset++] % 8) + 1;
            } else {
                tensor_shape[cat_dim] = (i % 4) + 1;
            }
            
            torch::Tensor tensor = torch::rand(tensor_shape, torch::dtype(dtype));
            tensors.push_back(tensor);
        }
        
        // Test torch::cat (same as torch::concat)
        torch::Tensor result = torch::cat(tensors, cat_dim);
        
        // Verify result shape
        int64_t expected_cat_size = 0;
        for (const auto& t : tensors) {
            expected_cat_size += t.size(cat_dim);
        }
        assert(result.size(cat_dim) == expected_cat_size);
        
        // Additional operations to increase coverage
        torch::Tensor sum = result.sum();
        
        // Test with TensorList
        torch::TensorList tensor_list(tensors);
        torch::Tensor result2 = torch::cat(tensor_list, cat_dim);
        
        // Test negative dimension
        if (offset < Size && Data[offset] % 2 == 0) {
            int64_t neg_dim = -(rank - cat_dim);
            torch::Tensor result_neg = torch::cat(tensors, neg_dim);
        }
        
        // Test with single tensor
        std::vector<torch::Tensor> single_tensor = {tensors[0]};
        torch::Tensor result_single = torch::cat(single_tensor, cat_dim);
        
        // Test contiguous path
        if (result.dim() > 0 && result.numel() > 0) {
            torch::Tensor contiguous = result.contiguous();
            torch::Tensor reshaped = result.reshape({-1});
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0;  // Keep the input
}