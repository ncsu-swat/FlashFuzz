#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        
        // Need at least 2 bytes to determine parameters
        if (Size < 2) {
            return 0;
        }
        
        // Determine number of tensors to concatenate (2-5)
        uint8_t num_tensors = (Data[offset++] % 4) + 2;
        
        // Determine dimension to concatenate along (0-3)
        int64_t dim = Data[offset++] % 4;
        
        // Create tensors with compatible shapes for concatenation
        std::vector<torch::Tensor> tensors;
        
        // Determine base shape from input data
        int64_t base_dim0 = 2, base_dim1 = 3, base_dim2 = 4;
        if (offset + 3 <= Size) {
            base_dim0 = (Data[offset++] % 4) + 1;
            base_dim1 = (Data[offset++] % 4) + 1;
            base_dim2 = (Data[offset++] % 4) + 1;
        }
        
        // Create tensors with shapes that are compatible for concatenation
        for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
            try {
                // Vary the size along the concatenation dimension
                int64_t cat_dim_size = (offset < Size) ? (Data[offset++] % 3) + 1 : 2;
                
                std::vector<int64_t> shape;
                if (dim == 0) {
                    shape = {cat_dim_size, base_dim1, base_dim2};
                } else if (dim == 1) {
                    shape = {base_dim0, cat_dim_size, base_dim2};
                } else if (dim == 2) {
                    shape = {base_dim0, base_dim1, cat_dim_size};
                } else {
                    shape = {base_dim0, base_dim1, base_dim2, cat_dim_size};
                }
                
                // Determine dtype
                auto dtype = torch::kFloat32;
                if (offset < Size) {
                    uint8_t dtype_sel = Data[offset++] % 4;
                    switch (dtype_sel) {
                        case 0: dtype = torch::kFloat32; break;
                        case 1: dtype = torch::kFloat64; break;
                        case 2: dtype = torch::kInt32; break;
                        case 3: dtype = torch::kInt64; break;
                    }
                }
                
                torch::Tensor tensor = torch::randn(shape, torch::TensorOptions().dtype(torch::kFloat32)).to(dtype);
                tensors.push_back(tensor);
            } catch (...) {
                // If we can't create a tensor, continue with what we have
                break;
            }
        }
        
        // Need at least two tensors to proceed meaningfully
        if (tensors.size() < 2) {
            return 0;
        }
        
        // Ensure all tensors have the same dtype for the main test
        auto target_dtype = tensors[0].dtype();
        for (size_t i = 1; i < tensors.size(); ++i) {
            tensors[i] = tensors[i].to(target_dtype);
        }
        
        // Apply torch.cat operation
        try {
            torch::Tensor result = torch::cat(tensors, dim);
            
            // Verify the result shape is correct
            (void)result.sizes();
        } catch (const c10::Error& e) {
            // PyTorch specific exceptions are expected for invalid inputs
        }
        
        // Test with dim=0 (should always work if shapes match otherwise)
        try {
            // Create tensors compatible for dim=0 concatenation
            std::vector<torch::Tensor> dim0_tensors;
            for (size_t i = 0; i < tensors.size(); ++i) {
                dim0_tensors.push_back(torch::randn({2, 3}, torch::kFloat32));
            }
            torch::Tensor result = torch::cat(dim0_tensors, 0);
        } catch (const c10::Error& e) {
            // Unexpected but handle gracefully
        }
        
        // Test with negative dimension
        try {
            torch::Tensor result = torch::cat(tensors, -1);
        } catch (const c10::Error& e) {
            // May fail if dimension is out of range
        }
        
        // Test with single tensor
        try {
            std::vector<torch::Tensor> single_tensor = {tensors[0]};
            torch::Tensor result = torch::cat(single_tensor, 0);
        } catch (const c10::Error& e) {
            // Should work
        }
        
        // Test with empty tensor list (expected to fail)
        try {
            std::vector<torch::Tensor> empty_tensors;
            torch::Tensor result = torch::cat(empty_tensors, 0);
        } catch (const c10::Error& e) {
            // Expected to fail
        }
        
        // Test with out parameter
        try {
            // Calculate expected output size
            int64_t total_cat_dim = 0;
            for (const auto& t : tensors) {
                if (dim < static_cast<int64_t>(t.dim())) {
                    total_cat_dim += t.size(dim);
                }
            }
            
            if (total_cat_dim > 0 && !tensors.empty()) {
                std::vector<int64_t> out_shape(tensors[0].sizes().begin(), tensors[0].sizes().end());
                if (dim < static_cast<int64_t>(out_shape.size())) {
                    out_shape[dim] = total_cat_dim;
                    auto options = torch::TensorOptions().dtype(tensors[0].dtype());
                    torch::Tensor out = torch::empty(out_shape, options);
                    torch::cat_out(out, tensors, dim);
                }
            }
        } catch (const c10::Error& e) {
            // Expected to fail in some cases
        }
        
        // Test with 1D tensors
        try {
            std::vector<torch::Tensor> tensors_1d;
            for (int i = 0; i < 3; ++i) {
                int64_t len = (offset < Size) ? (Data[offset++] % 5) + 1 : 3;
                tensors_1d.push_back(torch::randn({len}));
            }
            torch::Tensor result = torch::cat(tensors_1d, 0);
        } catch (const c10::Error& e) {
            // Handle gracefully
        }
        
        // Test with 2D tensors along dim=1
        try {
            std::vector<torch::Tensor> tensors_2d;
            int64_t rows = 3;
            for (int i = 0; i < 3; ++i) {
                int64_t cols = (offset < Size) ? (Data[offset++] % 4) + 1 : 2;
                tensors_2d.push_back(torch::randn({rows, cols}));
            }
            torch::Tensor result = torch::cat(tensors_2d, 1);
        } catch (const c10::Error& e) {
            // Handle gracefully
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}