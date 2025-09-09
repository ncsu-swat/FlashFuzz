#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Parse number of tensors to stack (1-10)
        uint8_t num_tensors = parse_uint8_t(Data, Size, offset) % 10 + 1;
        
        std::vector<torch::Tensor> tensors;
        
        for (uint8_t i = 0; i < num_tensors; i++) {
            // Parse tensor dimensions (1D or 2D for dstack)
            uint8_t ndim = parse_uint8_t(Data, Size, offset) % 2 + 1;
            
            std::vector<int64_t> shape;
            for (uint8_t j = 0; j < ndim; j++) {
                // Keep dimensions reasonable (1-20)
                int64_t dim_size = parse_uint8_t(Data, Size, offset) % 20 + 1;
                shape.push_back(dim_size);
            }
            
            // Parse tensor properties
            torch::ScalarType dtype = parse_dtype(Data, Size, offset);
            bool requires_grad = parse_bool(Data, Size, offset);
            
            // Create tensor with random data
            torch::Tensor tensor = torch::randn(shape, torch::TensorOptions().dtype(dtype).requires_grad(requires_grad));
            
            // Test edge cases with special values
            uint8_t special_case = parse_uint8_t(Data, Size, offset) % 10;
            if (special_case == 0) {
                tensor = torch::zeros_like(tensor);
            } else if (special_case == 1) {
                tensor = torch::ones_like(tensor);
            } else if (special_case == 2 && dtype.isFloatingPoint()) {
                tensor.fill_(std::numeric_limits<double>::infinity());
            } else if (special_case == 3 && dtype.isFloatingPoint()) {
                tensor.fill_(-std::numeric_limits<double>::infinity());
            } else if (special_case == 4 && dtype.isFloatingPoint()) {
                tensor.fill_(std::numeric_limits<double>::quiet_NaN());
            }
            
            tensors.push_back(tensor);
        }
        
        // Test torch::dstack with vector of tensors
        torch::Tensor result1 = torch::dstack(tensors);
        
        // Verify result properties
        if (result1.defined()) {
            // Check that result has correct number of dimensions
            if (result1.dim() >= 3) {
                // Verify the depth dimension matches number of tensors
                if (result1.size(2) == static_cast<int64_t>(num_tensors)) {
                    // Access some elements to trigger potential issues
                    if (result1.numel() > 0) {
                        auto flat = result1.flatten();
                        if (flat.numel() > 0) {
                            flat[0].item<double>();
                        }
                    }
                }
            }
        }
        
        // Test with TensorList (alternative interface)
        torch::TensorList tensor_list(tensors);
        torch::Tensor result2 = torch::dstack(tensor_list);
        
        // Test edge case: single tensor
        if (!tensors.empty()) {
            std::vector<torch::Tensor> single_tensor = {tensors[0]};
            torch::Tensor result3 = torch::dstack(single_tensor);
        }
        
        // Test with different device if CUDA is available
        if (torch::cuda::is_available() && parse_bool(Data, Size, offset)) {
            std::vector<torch::Tensor> cuda_tensors;
            for (const auto& tensor : tensors) {
                if (tensor.dtype() != torch::kBool) { // Bool tensors may not support CUDA
                    cuda_tensors.push_back(tensor.to(torch::kCUDA));
                }
            }
            if (!cuda_tensors.empty()) {
                torch::Tensor cuda_result = torch::dstack(cuda_tensors);
            }
        }
        
        // Test with mixed tensor properties (different requires_grad)
        if (tensors.size() >= 2) {
            std::vector<torch::Tensor> mixed_tensors;
            for (size_t i = 0; i < tensors.size(); i++) {
                auto tensor_copy = tensors[i].clone();
                tensor_copy.set_requires_grad(i % 2 == 0);
                mixed_tensors.push_back(tensor_copy);
            }
            torch::Tensor mixed_result = torch::dstack(mixed_tensors);
        }
        
        // Test backward pass if gradients are enabled
        if (result1.requires_grad() && result1.numel() > 0) {
            torch::Tensor loss = result1.sum();
            loss.backward();
        }
        
        // Test with empty tensor list (should throw)
        try {
            std::vector<torch::Tensor> empty_tensors;
            torch::Tensor empty_result = torch::dstack(empty_tensors);
        } catch (const std::exception&) {
            // Expected to throw
        }
        
        // Test memory layout variations
        if (!tensors.empty() && parse_bool(Data, Size, offset)) {
            std::vector<torch::Tensor> contiguous_tensors;
            for (const auto& tensor : tensors) {
                contiguous_tensors.push_back(tensor.contiguous());
            }
            torch::Tensor contiguous_result = torch::dstack(contiguous_tensors);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}