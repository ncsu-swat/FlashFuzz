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
        if (Size < 10) {
            return 0;
        }
        
        // Parse values tensor
        torch::Tensor values = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse crow_indices tensor
        torch::Tensor crow_indices;
        if (offset < Size) {
            crow_indices = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Parse col_indices tensor
        torch::Tensor col_indices;
        if (offset < Size) {
            col_indices = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Parse size parameter (optional)
        std::vector<int64_t> size_param;
        if (offset + 1 < Size) {
            uint8_t use_size = Data[offset++];
            if (use_size % 2 == 1) {  // 50% chance to use size parameter
                uint8_t size_rank = Data[offset++] % 3 + 2;  // 2-4 dimensions
                size_param = fuzzer_utils::parseShape(Data, offset, Size, size_rank);
            }
        }
        
        // Create sparse CSR tensor
        torch::Tensor sparse_tensor;
        if (size_param.empty()) {
            sparse_tensor = torch::sparse_csr_tensor(crow_indices, col_indices, values, torch::TensorOptions());
        } else {
            sparse_tensor = torch::sparse_csr_tensor(crow_indices, col_indices, values, size_param, torch::TensorOptions());
        }
        
        // Test some operations on the sparse tensor
        if (sparse_tensor.defined()) {
            auto dense = sparse_tensor.to_dense();
            auto indices = sparse_tensor.indices();
            auto values_extracted = sparse_tensor.values();
            auto sparse_dim = sparse_tensor.sparse_dim();
            auto dense_dim = sparse_tensor.dense_dim();
            auto is_coalesced = sparse_tensor.is_coalesced();
            
            // Try some math operations if possible
            try {
                if (sparse_tensor.dim() > 0) {
                    auto sum = sparse_tensor.sum();
                }
                
                // Try matrix multiplication if shape allows
                if (sparse_tensor.dim() >= 2) {
                    auto vector_size = sparse_tensor.size(1);
                    auto vector = torch::ones({vector_size});
                    try {
                        auto result = sparse_tensor.matmul(vector);
                    } catch (...) {
                        // Ignore errors from matmul
                    }
                }
            } catch (...) {
                // Ignore errors from operations
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
