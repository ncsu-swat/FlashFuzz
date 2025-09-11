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
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Parse indices tensor
        torch::Tensor indices;
        if (offset < Size) {
            indices = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Parse values tensor
        torch::Tensor values;
        if (offset < Size) {
            values = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Parse sparse dimensions
        int64_t sparse_dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&sparse_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Allow negative values to test error handling
        }
        
        // Parse dense dimensions
        int64_t dense_dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dense_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Allow negative values to test error handling
        }
        
        // Parse size (shape) for the sparse tensor
        std::vector<int64_t> size;
        uint8_t num_dims = 0;
        if (offset < Size) {
            num_dims = Data[offset++] % 5; // 0-4 dimensions
            
            for (uint8_t i = 0; i < num_dims && offset + sizeof(int64_t) <= Size; i++) {
                int64_t dim_size;
                std::memcpy(&dim_size, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                size.push_back(dim_size); // Allow negative values to test error handling
            }
        }
        
        // Parse dtype
        torch::ScalarType dtype = torch::kFloat;
        if (offset < Size) {
            dtype = fuzzer_utils::parseDataType(Data[offset++]);
        }
        
        // Try different variants of sparse_coo_tensor
        try {
            // Variant 1: Basic sparse_coo_tensor with indices and values
            torch::Tensor sparse1 = torch::sparse_coo_tensor(indices, values);
            
            // Variant 2: With size parameter
            if (!size.empty()) {
                torch::Tensor sparse2 = torch::sparse_coo_tensor(indices, values, size);
            }
            
            // Variant 3: With dtype
            torch::Tensor sparse3 = torch::sparse_coo_tensor(
                indices, values, torch::TensorOptions().dtype(dtype));
            
            // Variant 4: With size and dtype
            if (!size.empty()) {
                torch::Tensor sparse4 = torch::sparse_coo_tensor(
                    indices, values, size, torch::TensorOptions().dtype(dtype));
            }
            
            // Test some operations on the sparse tensor
            if (!size.empty()) {
                torch::Tensor sparse = torch::sparse_coo_tensor(indices, values, size);
                
                // Test coalesce
                if (!sparse.is_coalesced()) {
                    torch::Tensor coalesced = sparse.coalesce();
                }
                
                // Test to_dense
                torch::Tensor dense = sparse.to_dense();
                
                // Test sparse operations
                if (sparse.dim() > 0) {
                    torch::Tensor sparse_sum = sparse.sum();
                }
            }
            
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected and ok
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
