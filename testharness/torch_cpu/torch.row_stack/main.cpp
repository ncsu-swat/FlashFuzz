#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Determine number of tensors to create (1-4)
        uint8_t num_tensors = (Data[0] % 4) + 1;
        offset++;
        
        // Create a vector to hold our tensors
        std::vector<torch::Tensor> tensors;
        
        // Create tensors
        for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                tensors.push_back(tensor);
            } catch (...) {
                // If we can't create a tensor, just continue with what we have
                break;
            }
        }
        
        // Need at least one tensor to proceed
        if (tensors.empty()) {
            return 0;
        }
        
        // Apply row_stack operation (equivalent to vstack/vertical stack)
        try {
            torch::Tensor result = torch::row_stack(tensors);
            // Use the result to prevent optimization
            (void)result.numel();
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected (shape mismatches, etc.)
        }
        
        // Try alternative syntax with vstack
        try {
            torch::Tensor result = torch::vstack(tensors);
            (void)result.numel();
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected
        }
        
        // Try with empty tensor list
        try {
            std::vector<torch::Tensor> empty_tensors;
            torch::Tensor result = torch::row_stack(empty_tensors);
            (void)result.numel();
        } catch (const c10::Error& e) {
            // Expected exception for empty list
        }
        
        // Try with tensors of different dtypes if we have multiple tensors
        if (tensors.size() > 1 && offset < Size) {
            try {
                uint8_t dtype_selector = Data[offset++];
                torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
                
                // Create copies to avoid modifying original tensors
                std::vector<torch::Tensor> mixed_tensors;
                mixed_tensors.push_back(tensors[0].to(dtype));
                for (size_t i = 1; i < tensors.size(); ++i) {
                    mixed_tensors.push_back(tensors[i]);
                }
                
                torch::Tensor result = torch::row_stack(mixed_tensors);
                (void)result.numel();
            } catch (const c10::Error& e) {
                // Expected exception for dtype mismatch
            }
        }
        
        // Try with tensors that have incompatible shapes
        try {
            torch::Tensor t1 = torch::ones({2, 3});
            torch::Tensor t2 = torch::ones({3, 4});
            
            std::vector<torch::Tensor> incompatible_tensors = {t1, t2};
            torch::Tensor result = torch::row_stack(incompatible_tensors);
            (void)result.numel();
        } catch (const c10::Error& e) {
            // Expected exception for shape mismatch
        }
        
        // Try with scalar tensors (0-D)
        try {
            std::vector<torch::Tensor> scalar_tensors;
            scalar_tensors.push_back(torch::tensor(1.0));
            scalar_tensors.push_back(torch::tensor(2.0));
            
            torch::Tensor result = torch::row_stack(scalar_tensors);
            (void)result.numel();
        } catch (const c10::Error& e) {
            // Expected - scalars get treated as 1D
        }
        
        // Try with 1D tensors
        try {
            std::vector<torch::Tensor> vector_tensors;
            vector_tensors.push_back(torch::ones({3}));
            vector_tensors.push_back(torch::ones({3}));
            
            torch::Tensor result = torch::row_stack(vector_tensors);
            (void)result.numel();
        } catch (const c10::Error& e) {
            // Handle any errors
        }
        
        // Try with 3D tensors
        try {
            std::vector<torch::Tensor> tensors_3d;
            tensors_3d.push_back(torch::ones({2, 3, 4}));
            tensors_3d.push_back(torch::ones({2, 3, 4}));
            
            torch::Tensor result = torch::row_stack(tensors_3d);
            (void)result.numel();
        } catch (const c10::Error& e) {
            // Handle any errors
        }
        
        // Try with single tensor
        try {
            std::vector<torch::Tensor> single_tensor;
            single_tensor.push_back(torch::ones({2, 3}));
            
            torch::Tensor result = torch::row_stack(single_tensor);
            (void)result.numel();
        } catch (const c10::Error& e) {
            // Handle any errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}