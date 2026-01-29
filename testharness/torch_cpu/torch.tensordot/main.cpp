#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        // Need at least some data to proceed
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
            // If no data left, create a simple tensor
            tensor2 = torch::ones({2, 3, 4});
        }
        
        // Skip if either tensor is 0-dimensional
        if (tensor1.dim() == 0 || tensor2.dim() == 0) {
            return 0;
        }
        
        // Extract dims parameter for tensordot
        int64_t dims = 0;
        if (offset + sizeof(uint8_t) <= Size) {
            dims = Data[offset++] % 4;  // Limit to 0-3 dimensions
        }
        
        // Variant 1: tensordot with matching last dims of tensor1 and first dims of tensor2
        {
            int64_t max_contract_dims = std::min(tensor1.dim(), tensor2.dim());
            int64_t num_contract_dims = std::min(max_contract_dims, dims);
            
            if (num_contract_dims > 0) {
                std::vector<int64_t> dims1, dims2;
                
                // Contract last num_contract_dims of tensor1 with first num_contract_dims of tensor2
                for (int64_t i = 0; i < num_contract_dims; i++) {
                    dims1.push_back(tensor1.dim() - num_contract_dims + i);
                    dims2.push_back(i);
                }
                
                try {
                    torch::Tensor result1 = torch::tensordot(tensor1, tensor2, dims1, dims2);
                } catch (...) {
                    // Expected for incompatible dimension sizes
                }
            }
        }
        
        // Variant 2: tensordot with fuzzer-controlled dimension indices
        if (offset + 2 <= Size) {
            int64_t num_dims_to_contract = (Data[offset++] % 3) + 1;  // 1-3 dims
            num_dims_to_contract = std::min(num_dims_to_contract, std::min(tensor1.dim(), tensor2.dim()));
            
            std::vector<int64_t> dims1, dims2;
            
            for (int64_t i = 0; i < num_dims_to_contract && offset + 2 <= Size; i++) {
                int64_t dim1 = Data[offset++] % tensor1.dim();
                int64_t dim2 = Data[offset++] % tensor2.dim();
                
                // Avoid duplicate dimensions
                bool dup1 = false, dup2 = false;
                for (auto d : dims1) if (d == dim1) dup1 = true;
                for (auto d : dims2) if (d == dim2) dup2 = true;
                
                if (!dup1 && !dup2) {
                    dims1.push_back(dim1);
                    dims2.push_back(dim2);
                }
            }
            
            if (!dims1.empty() && dims1.size() == dims2.size()) {
                try {
                    torch::Tensor result2 = torch::tensordot(tensor1, tensor2, dims1, dims2);
                } catch (...) {
                    // Expected for incompatible dimension sizes
                }
            }
        }
        
        // Variant 3: Try with different data types
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            try {
                torch::Tensor tensor1_cast = tensor1.to(dtype);
                torch::Tensor tensor2_cast = tensor2.to(dtype);
                
                // Simple contraction: last dim of tensor1 with first dim of tensor2
                if (tensor1_cast.dim() > 0 && tensor2_cast.dim() > 0) {
                    std::vector<int64_t> d1 = {tensor1_cast.dim() - 1};
                    std::vector<int64_t> d2 = {0};
                    torch::Tensor result3 = torch::tensordot(tensor1_cast, tensor2_cast, d1, d2);
                }
            } catch (...) {
                // Some dtype combinations might be invalid
            }
        }
        
        // Variant 4: Empty contraction (outer product)
        try {
            std::vector<int64_t> empty_dims1, empty_dims2;
            torch::Tensor outer_product = torch::tensordot(tensor1, tensor2, empty_dims1, empty_dims2);
        } catch (...) {
            // May fail for very large tensors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}