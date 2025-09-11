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
            tensor2 = torch::ones({1, 2, 3});
        }
        
        // Extract dims parameter for tensordot
        int64_t dims = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dims, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure dims is within a reasonable range
            dims = std::abs(dims) % 5;
        }
        
        // Try different variants of tensordot
        
        // Variant 1: Basic tensordot with scalar dims - create dimension arrays
        if (tensor1.dim() > 0 && tensor2.dim() > 0) {
            std::vector<int64_t> dims1_basic, dims2_basic;
            int64_t max_contract_dims = std::min(tensor1.dim(), tensor2.dim());
            int64_t num_contract_dims = std::min(max_contract_dims, dims);
            
            for (int64_t i = 0; i < num_contract_dims; i++) {
                dims1_basic.push_back(tensor1.dim() - 1 - i);
                dims2_basic.push_back(i);
            }
            
            if (!dims1_basic.empty() && !dims2_basic.empty()) {
                try {
                    torch::Tensor result1 = torch::tensordot(tensor1, tensor2, dims1_basic, dims2_basic);
                } catch (const std::exception& e) {
                    // Catch and continue - this is expected for some incompatible dimensions
                }
            }
        }
        
        // Variant 2: tensordot with list of dimensions
        if (tensor1.dim() > 0 && tensor2.dim() > 0) {
            std::vector<int64_t> dims1, dims2;
            
            // Extract dimensions for contraction
            int64_t max_contract_dims = std::min(tensor1.dim(), tensor2.dim());
            int64_t num_contract_dims = std::min(max_contract_dims, dims);
            
            for (int64_t i = 0; i < num_contract_dims; i++) {
                int64_t dim1 = 0, dim2 = 0;
                
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&dim1, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    dim1 = std::abs(dim1) % tensor1.dim();
                }
                
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&dim2, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    dim2 = std::abs(dim2) % tensor2.dim();
                }
                
                dims1.push_back(dim1);
                dims2.push_back(dim2);
            }
            
            // Try tensordot with dimension lists
            if (!dims1.empty() && !dims2.empty()) {
                try {
                    torch::Tensor result2 = torch::tensordot(tensor1, tensor2, dims1, dims2);
                } catch (const std::exception& e) {
                    // Catch and continue - this is expected for some incompatible dimensions
                }
            }
        }
        
        // Variant 3: Try with different data types
        if (offset < Size && tensor1.dim() > 0 && tensor2.dim() > 0) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            try {
                torch::Tensor tensor1_cast = tensor1.to(dtype);
                torch::Tensor tensor2_cast = tensor2.to(dtype);
                
                std::vector<int64_t> dims1_cast, dims2_cast;
                int64_t max_contract_dims = std::min(tensor1_cast.dim(), tensor2_cast.dim());
                int64_t num_contract_dims = std::min(max_contract_dims, dims);
                
                for (int64_t i = 0; i < num_contract_dims; i++) {
                    dims1_cast.push_back(tensor1_cast.dim() - 1 - i);
                    dims2_cast.push_back(i);
                }
                
                if (!dims1_cast.empty() && !dims2_cast.empty()) {
                    torch::Tensor result3 = torch::tensordot(tensor1_cast, tensor2_cast, dims1_cast, dims2_cast);
                }
            } catch (const std::exception& e) {
                // Catch and continue - some dtype combinations might be invalid
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
