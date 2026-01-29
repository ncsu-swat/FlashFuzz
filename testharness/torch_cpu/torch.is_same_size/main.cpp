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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create first tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second tensor if there's data left
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no data left, create a tensor with different shape
            auto options = torch::TensorOptions().dtype(tensor1.dtype());
            std::vector<int64_t> shape;
            
            // Create a shape that's different from tensor1
            if (tensor1.dim() > 0) {
                shape = tensor1.sizes().vec();
                if (shape.size() > 0) {
                    // Modify one dimension to make it different
                    shape[0] = shape[0] + 1;
                }
            } else {
                // If tensor1 is a scalar, create a 1D tensor
                shape.push_back(1);
            }
            
            tensor2 = torch::empty(shape, options);
        }
        
        // Apply is_same_size operation
        bool result = tensor1.sizes() == tensor2.sizes();
        
        // Alternative way to check using PyTorch C++ API
        bool api_result = at::is_same_size(tensor1, tensor2);
        
        // Verify that both methods give the same result
        if (result != api_result) {
            throw std::runtime_error("Inconsistent results between manual check and at::is_same_size");
        }
        
        // Test edge cases with empty tensors
        if (Size > offset + 2) {
            uint8_t edge_case = Data[offset++];
            
            if (edge_case % 4 == 0) {
                // Test with empty tensors of same shape
                torch::Tensor empty1 = torch::empty({0});
                torch::Tensor empty2 = torch::empty({0});
                bool empty_result = at::is_same_size(empty1, empty2);
                
                // Empty tensors with same shape should have the same size
                if (!empty_result) {
                    throw std::runtime_error("Empty tensors with same shape should have the same size");
                }
                
                // Test empty tensors with different shapes
                torch::Tensor empty3 = torch::empty({0, 1});
                bool diff_empty_result = at::is_same_size(empty1, empty3);
                if (diff_empty_result) {
                    throw std::runtime_error("Empty tensors with different shapes should not have the same size");
                }
            } else if (edge_case % 4 == 1) {
                // Test with tensors of same shape (should be true)
                if (tensor1.dim() > 0 && tensor1.numel() > 0) {
                    torch::Tensor same_shape = torch::empty(tensor1.sizes().vec(), tensor1.options());
                    bool same_result = at::is_same_size(tensor1, same_shape);
                    if (!same_result) {
                        throw std::runtime_error("Tensors with same shape should have the same size");
                    }
                }
            } else if (edge_case % 4 == 2) {
                // Test with tensors of different dimensions but same number of elements
                if (tensor1.dim() > 1 && tensor1.numel() > 0) {
                    std::vector<int64_t> new_shape;
                    int64_t total_elements = tensor1.numel();
                    
                    // Create a 1D shape with the same number of elements
                    new_shape.push_back(total_elements);
                    
                    torch::Tensor reshaped = torch::empty(new_shape, tensor1.options());
                    bool reshape_result = at::is_same_size(tensor1, reshaped);
                    
                    // Should be false since shapes are different
                    if (reshape_result) {
                        throw std::runtime_error("Differently shaped tensors should not have the same size");
                    }
                }
            } else {
                // Test with scalar tensors
                torch::Tensor scalar1 = torch::tensor(1.0);
                torch::Tensor scalar2 = torch::tensor(2.0);
                bool scalar_result = at::is_same_size(scalar1, scalar2);
                
                // Scalar tensors (0-dim) should have the same size
                if (!scalar_result) {
                    throw std::runtime_error("Scalar tensors should have the same size");
                }
                
                // Scalar vs 1D tensor should be different
                torch::Tensor one_d = torch::tensor({1.0});
                bool scalar_vs_1d = at::is_same_size(scalar1, one_d);
                if (scalar_vs_1d) {
                    throw std::runtime_error("Scalar and 1D tensor should not have the same size");
                }
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