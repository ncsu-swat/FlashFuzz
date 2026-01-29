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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract dimension parameter from fuzzer data
        int64_t dim_raw = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Skip scalar tensors - unbind_copy requires at least 1 dimension
        if (input_tensor.dim() == 0) {
            return 0;
        }
        
        // Test 1: unbind_copy with bounded dimension
        {
            // Bound dimension to valid range
            int64_t dim = dim_raw % input_tensor.dim();
            
            try {
                std::vector<torch::Tensor> result = torch::unbind_copy(input_tensor, dim);
                
                // Perform operations on resulting tensors to ensure they're valid
                for (const auto& tensor : result) {
                    auto sum = tensor.sum();
                    (void)sum;
                }
            } catch (const c10::Error& e) {
                // Silently catch shape/dim errors
            }
        }
        
        // Test 2: unbind_copy with default dimension (0)
        {
            try {
                std::vector<torch::Tensor> result_default = torch::unbind_copy(input_tensor);
                
                // Perform operations on resulting tensors
                for (const auto& tensor : result_default) {
                    auto sum = tensor.sum();
                    (void)sum;
                }
            } catch (const c10::Error& e) {
                // Silently catch errors
            }
        }
        
        // Test 3: unbind_copy with negative dimension (valid in PyTorch)
        {
            try {
                int64_t neg_dim = -(std::abs(dim_raw) % input_tensor.dim()) - 1;
                std::vector<torch::Tensor> result_neg = torch::unbind_copy(input_tensor, neg_dim);
                
                for (const auto& tensor : result_neg) {
                    auto sum = tensor.sum();
                    (void)sum;
                }
            } catch (const c10::Error& e) {
                // Silently catch errors for invalid dims
            }
        }
        
        // Test 4: Test with different tensor types if we have more fuzzer data
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset] % 4;
            offset++;
            
            torch::Tensor typed_tensor;
            try {
                switch (dtype_selector) {
                    case 0:
                        typed_tensor = input_tensor.to(torch::kFloat32);
                        break;
                    case 1:
                        typed_tensor = input_tensor.to(torch::kFloat64);
                        break;
                    case 2:
                        typed_tensor = input_tensor.to(torch::kInt32);
                        break;
                    case 3:
                        typed_tensor = input_tensor.to(torch::kInt64);
                        break;
                }
                
                if (typed_tensor.dim() > 0) {
                    int64_t dim = dim_raw % typed_tensor.dim();
                    std::vector<torch::Tensor> result = torch::unbind_copy(typed_tensor, dim);
                    
                    for (const auto& tensor : result) {
                        auto mean = tensor.mean();
                        (void)mean;
                    }
                }
            } catch (const c10::Error& e) {
                // Silently catch type conversion or unbind errors
            }
        }
        
        // Test 5: Test with contiguous vs non-contiguous tensor
        {
            try {
                if (input_tensor.dim() >= 2) {
                    // Create a non-contiguous view via transpose
                    torch::Tensor transposed = input_tensor.transpose(0, input_tensor.dim() - 1);
                    int64_t dim = dim_raw % transposed.dim();
                    
                    std::vector<torch::Tensor> result = torch::unbind_copy(transposed, dim);
                    
                    for (const auto& tensor : result) {
                        auto sum = tensor.sum();
                        (void)sum;
                    }
                }
            } catch (const c10::Error& e) {
                // Silently catch errors
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