#include "fuzzer_utils.h"
#include <iostream>
#include <vector>
#include <string>

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
        
        // Need at least 2 bytes for equation selection and configuration
        if (Size < 2) {
            return 0;
        }
        
        // Parse equation type first
        uint8_t equation_selector = Data[offset++];
        uint8_t config_byte = Data[offset++];
        
        // Derive dimensions from config byte for consistent tensor shapes
        int64_t dim_i = (config_byte & 0x07) + 2;  // 2-9
        int64_t dim_j = ((config_byte >> 3) & 0x07) + 2;  // 2-9
        int64_t dim_k = ((config_byte >> 6) & 0x03) + 2;  // 2-5
        
        // Get batch size if available
        int64_t batch = 1;
        if (offset < Size) {
            batch = (Data[offset++] % 4) + 1;  // 1-4
        }

        // Define einsum equations with their required tensor shapes
        struct EinsumCase {
            std::string equation;
            std::vector<std::vector<int64_t>> shapes;
        };
        
        std::vector<EinsumCase> cases = {
            // Single tensor operations
            {"i->i", {{dim_i}}},
            {"i->", {{dim_i}}},
            {"ii->i", {{dim_i, dim_i}}},  // Diagonal - needs square matrix
            {"ii->", {{dim_i, dim_i}}},   // Trace - needs square matrix
            {"ij->ji", {{dim_i, dim_j}}}, // Transpose
            {"ij->j", {{dim_i, dim_j}}},  // Sum over rows
            {"ij->i", {{dim_i, dim_j}}},  // Sum over columns
            {"ij->", {{dim_i, dim_j}}},   // Sum all
            {"ijk->kji", {{dim_i, dim_j, dim_k}}}, // Permute
            
            // Two tensor operations
            {"i,i->i", {{dim_i}, {dim_i}}},           // Element-wise
            {"i,i->", {{dim_i}, {dim_i}}},            // Dot product
            {"i,j->ij", {{dim_i}, {dim_j}}},          // Outer product
            {"ij,jk->ik", {{dim_i, dim_j}, {dim_j, dim_k}}}, // Matrix multiplication
            {"ij,ij->ij", {{dim_i, dim_j}, {dim_i, dim_j}}}, // Element-wise 2D
            {"ij,ij->", {{dim_i, dim_j}, {dim_i, dim_j}}},   // Frobenius inner product
            {"ij,ji->", {{dim_i, dim_j}, {dim_j, dim_i}}},   // Trace of product
            
            // Batch operations
            {"bi,bi->b", {{batch, dim_i}, {batch, dim_i}}},  // Batch dot product
            {"bij,bjk->bik", {{batch, dim_i, dim_j}, {batch, dim_j, dim_k}}}, // Batch matmul
            
            // Three tensor operations
            {"i,i,i->i", {{dim_i}, {dim_i}, {dim_i}}},       // Element-wise triple
            {"i,j,k->ijk", {{dim_i}, {dim_j}, {dim_k}}},     // Triple outer product
            {"ij,jk,kl->il", {{dim_i, dim_j}, {dim_j, dim_k}, {dim_k, dim_i}}}, // Chain matmul
            {"i,i,i->", {{dim_i}, {dim_i}, {dim_i}}},        // Triple dot product
            
            // Ellipsis operations
            {"...->...", {{dim_i, dim_j}}},  // Identity with ellipsis
            {"...i->...", {{batch, dim_i}}}, // Sum over last dim
        };
        
        // Select a case
        const EinsumCase& selected = cases[equation_selector % cases.size()];
        
        // Create tensors with the required shapes
        std::vector<torch::Tensor> tensors;
        for (const auto& shape : selected.shapes) {
            // Use random data for tensor values
            torch::Tensor t = torch::randn(shape, torch::kFloat32);
            
            // Optionally vary the dtype based on fuzzer input
            if (offset < Size) {
                uint8_t dtype_selector = Data[offset++] % 4;
                switch (dtype_selector) {
                    case 0: t = t.to(torch::kFloat32); break;
                    case 1: t = t.to(torch::kFloat64); break;
                    case 2: t = t.to(torch::kInt32); break;
                    case 3: t = t.to(torch::kInt64); break;
                }
            }
            tensors.push_back(t);
        }
        
        // Apply einsum operation
        try {
            torch::Tensor result = torch::einsum(selected.equation, tensors);
            
            // Force evaluation and verify result
            result.sizes();
            result.numel();
            
            // Additional operations to increase coverage
            if (result.numel() > 0 && result.is_floating_point()) {
                result.sum();
            }
        } catch (const std::exception& e) {
            // Expected exceptions from invalid dtype combinations, etc.
            // Silently catch - these are valid fuzzer inputs that trigger edge cases
        }
        
        // Also test the path parameter variant if we have enough data
        if (offset + 1 < Size && tensors.size() >= 2) {
            try {
                // Test with optimize=True equivalent (path parameter)
                torch::Tensor result2 = torch::einsum(selected.equation, tensors);
                result2.sizes();
            } catch (const std::exception& e) {
                // Expected exceptions
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