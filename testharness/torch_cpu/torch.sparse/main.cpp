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
        
        // Need at least a few bytes to create meaningful tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create a dense tensor to convert to sparse
        torch::Tensor dense_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create indices and values for sparse tensor
        torch::Tensor indices;
        torch::Tensor values;
        
        // Try different sparse tensor creation approaches based on available data
        if (offset + 2 < Size) {
            uint8_t sparse_format = Data[offset++];
            
            // COO format sparse tensor
            if (sparse_format % 2 == 0) {
                // Create a sparse tensor from the dense tensor
                torch::Tensor sparse_tensor = dense_tensor.to_sparse();
                
                // Test basic operations on sparse tensor
                auto sparse_size = sparse_tensor.sizes();
                auto sparse_indices = sparse_tensor.indices();
                auto sparse_values = sparse_tensor.values();
                
                // Convert back to dense
                torch::Tensor dense_again = sparse_tensor.to_dense();
                
                // Test sparse operations
                if (sparse_tensor.dim() > 0) {
                    int64_t dim = Data[offset % sparse_tensor.dim()];
                    if (offset < Size) offset++;
                    
                    // Test coalesce
                    if (!sparse_tensor.is_coalesced()) {
                        torch::Tensor coalesced = sparse_tensor.coalesce();
                    }
                }
            } 
            // Create sparse tensor directly
            else {
                // Create indices tensor
                if (offset < Size) {
                    indices = fuzzer_utils::createTensor(Data, Size, offset);
                    
                    // Create values tensor if we have more data
                    if (offset < Size) {
                        values = fuzzer_utils::createTensor(Data, Size, offset);
                        
                        // Get sparse dimensions
                        std::vector<int64_t> sparse_dims;
                        if (dense_tensor.dim() > 0) {
                            for (int i = 0; i < dense_tensor.dim(); i++) {
                                sparse_dims.push_back(dense_tensor.size(i));
                            }
                        } else {
                            // Default dimensions if dense tensor is a scalar
                            sparse_dims = {2, 3};
                        }
                        
                        // Try to create a sparse tensor
                        try {
                            torch::Tensor sparse_tensor = torch::sparse_coo_tensor(
                                indices, values, sparse_dims);
                                
                            // Test basic operations
                            auto dense_version = sparse_tensor.to_dense();
                            
                            // Test coalesce if not already coalesced
                            if (!sparse_tensor.is_coalesced()) {
                                auto coalesced = sparse_tensor.coalesce();
                            }
                            
                            // Test sparse operations
                            if (offset + 1 < Size) {
                                uint8_t op_type = Data[offset++];
                                
                                // Test sparse add
                                if (op_type % 4 == 0) {
                                    torch::Tensor sparse_tensor2 = dense_tensor.to_sparse();
                                    torch::Tensor result = sparse_tensor + sparse_tensor2;
                                }
                                // Test sparse mul
                                else if (op_type % 4 == 1) {
                                    torch::Tensor scalar = torch::tensor(2.0);
                                    torch::Tensor result = sparse_tensor * scalar;
                                }
                                // Test sparse transpose
                                else if (op_type % 4 == 2 && sparse_tensor.dim() >= 2) {
                                    torch::Tensor result = sparse_tensor.transpose(0, 1);
                                }
                                // Test sparse mm
                                else if (op_type % 4 == 3 && sparse_tensor.dim() == 2) {
                                    torch::Tensor dense_mat = torch::rand({sparse_tensor.size(1), 5});
                                    torch::Tensor result = torch::mm(sparse_tensor, dense_mat);
                                }
                            }
                        } catch (const c10::Error& e) {
                            // Catch PyTorch-specific errors during sparse tensor creation
                        }
                    }
                }
            }
        }
        
        // Test sparse_mask operation
        if (offset + 1 < Size) {
            try {
                torch::Tensor mask = dense_tensor.gt(0).to_sparse();
                torch::Tensor result = dense_tensor.sparse_mask(mask);
            } catch (const c10::Error& e) {
                // Catch PyTorch-specific errors
            }
        }
        
        // Test sparse resize operations
        if (offset + 1 < Size && dense_tensor.dim() > 0) {
            try {
                torch::Tensor sparse_tensor = dense_tensor.to_sparse();
                
                // Get new size
                int64_t new_size = static_cast<int64_t>(Data[offset++]) % 10 + 1;
                std::vector<int64_t> new_sizes = sparse_tensor.sizes().vec();
                if (!new_sizes.empty()) {
                    new_sizes[0] = new_size;
                    sparse_tensor.sparse_resize_(new_sizes, sparse_tensor.sparse_dim(), sparse_tensor.dense_dim());
                }
            } catch (const c10::Error& e) {
                // Catch PyTorch-specific errors
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
