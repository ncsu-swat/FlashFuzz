#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Determine number of tensors to use (2-5)
        if (Size < 1) return 0;
        uint8_t num_tensors = (Data[offset++] % 4) + 2; // 2 to 5 tensors
        
        // Create a vector to store our tensors
        std::vector<torch::Tensor> tensors;
        
        // Create tensors for chain_matmul
        for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // For chain_matmul, tensors must be 2D matrices
            // If tensor is not 2D, reshape it to make it 2D
            if (tensor.dim() != 2) {
                int64_t total_elements = tensor.numel();
                
                // Create a valid 2D shape
                int64_t dim1 = 1;
                int64_t dim2 = total_elements;
                
                // If we have more data, use it to determine dimensions
                if (offset + 2 < Size) {
                    dim1 = (Data[offset++] % 8) + 1; // 1-8
                    
                    // Ensure dim2 is valid
                    if (total_elements > 0) {
                        dim2 = (total_elements + dim1 - 1) / dim1; // Ceiling division
                        
                        // Adjust dim1 to ensure dim1*dim2 >= total_elements
                        int64_t new_total = dim1 * dim2;
                        if (new_total > total_elements) {
                            // Need to pad the tensor
                            tensor = tensor.reshape({-1});
                            tensor = torch::pad(tensor, {0, new_total - total_elements});
                        }
                    } else {
                        dim2 = (Data[offset++] % 8) + 1; // 1-8 for empty tensor
                    }
                }
                
                // Reshape to 2D
                tensor = tensor.reshape({dim1, dim2});
            }
            
            // For chain_matmul, ensure dimensions are compatible
            if (i > 0) {
                // The inner dimensions must match: A(m,n) * B(n,p) = C(m,p)
                // So tensor[i-1].size(1) must equal tensor[i].size(0)
                torch::Tensor& prev_tensor = tensors.back();
                
                if (prev_tensor.size(1) != tensor.size(0) && 
                    prev_tensor.numel() > 0 && tensor.numel() > 0) {
                    // Reshape current tensor to make dimensions compatible
                    tensor = tensor.reshape({prev_tensor.size(1), -1});
                }
            }
            
            tensors.push_back(tensor);
        }
        
        // Ensure we have at least 2 tensors
        if (tensors.size() < 2) {
            if (tensors.empty()) {
                // Create two default tensors if none were created
                tensors.push_back(torch::ones({2, 3}));
                tensors.push_back(torch::ones({3, 2}));
            } else {
                // Create a compatible second tensor
                auto shape = tensors[0].sizes();
                tensors.push_back(torch::ones({shape[1], shape[0]}));
            }
        }
        
        // Apply chain_matmul operation
        torch::Tensor result;
        try {
            result = torch::chain_matmul(tensors);
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected and not a bug in our fuzzer
            return 0;
        }
        
        // Verify result is not empty and has expected shape
        if (result.numel() > 0) {
            auto first_tensor = tensors.front();
            auto last_tensor = tensors.back();
            
            // The result should have shape [first_tensor.size(0), last_tensor.size(1)]
            if (first_tensor.dim() > 0 && last_tensor.dim() > 0) {
                int64_t expected_rows = first_tensor.size(0);
                int64_t expected_cols = last_tensor.size(1);
                
                if (result.size(0) != expected_rows || result.size(1) != expected_cols) {
                    // This would indicate a bug in PyTorch's implementation
                    throw std::runtime_error("Unexpected result shape");
                }
            }
        }
        
        // Test some operations on the result to ensure it's valid
        if (result.numel() > 0) {
            auto sum = result.sum();
            auto mean = result.mean();
            auto max_val = result.max();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
