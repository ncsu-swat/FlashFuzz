#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 8) {
            return 0;
        }
        
        // Parse compression format (CSR, CSC, BSR, BSC)
        uint8_t format_selector = Data[offset++];
        std::string layout;
        switch (format_selector % 4) {
            case 0: layout = "csr"; break;
            case 1: layout = "csc"; break;
            case 2: layout = "bsr"; break;
            case 3: layout = "bsc"; break;
        }
        
        // Create values tensor
        torch::Tensor values = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create compressed indices tensor (int32 or int64)
        torch::Tensor compressed_indices;
        if (offset < Size) {
            compressed_indices = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Default to a small tensor if we've run out of data
            compressed_indices = torch::tensor({0, 1}, torch::kInt64);
        }
        
        // Create plain indices tensor (int32 or int64)
        torch::Tensor plain_indices;
        if (offset < Size) {
            plain_indices = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Default to a small tensor if we've run out of data
            plain_indices = torch::tensor({0}, torch::kInt64);
        }
        
        // Parse size parameter
        std::vector<int64_t> size_vec;
        if (offset + 2 < Size) {
            uint8_t size_rank = Data[offset++] % 3 + 1; // 1-3 dimensions
            for (int i = 0; i < size_rank; i++) {
                if (offset + sizeof(int64_t) <= Size) {
                    int64_t dim_size;
                    std::memcpy(&dim_size, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    // Allow negative sizes to test error handling
                    size_vec.push_back(dim_size);
                } else {
                    size_vec.push_back(10); // Default size
                }
            }
        } else {
            // Default size if not enough data
            size_vec = {10, 10};
        }
        
        // Try to create sparse compressed tensor with various parameters
        try {
            torch::Tensor sparse_tensor = torch::sparse_compressed_tensor(
                compressed_indices,
                plain_indices,
                values,
                c10::IntArrayRef(size_vec),
                torch::TensorOptions().layout(torch::kSparse).dtype(values.dtype())
            );
            
            // Test some operations on the sparse tensor
            if (sparse_tensor.defined()) {
                auto dense = sparse_tensor.to_dense();
                auto indices = sparse_tensor._indices();
                auto values_extracted = sparse_tensor._values();
            }
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected and okay
        }
        
        // Try with different layout options
        if (offset < Size) {
            try {
                torch::Tensor sparse_tensor2 = torch::sparse_compressed_tensor(
                    compressed_indices,
                    plain_indices,
                    values,
                    torch::TensorOptions().dtype(values.dtype())
                );
                
                // Test some operations on the sparse tensor
                if (sparse_tensor2.defined()) {
                    auto dense = sparse_tensor2.to_dense();
                }
            } catch (const c10::Error& e) {
                // PyTorch specific errors are expected and okay
            }
        }
        
        // Try with empty tensors
        if (offset < Size) {
            try {
                torch::Tensor empty_values = torch::empty({0}, values.options());
                torch::Tensor empty_compressed = torch::tensor({0}, compressed_indices.options());
                torch::Tensor empty_plain = torch::tensor({}, plain_indices.options());
                
                torch::Tensor sparse_tensor3 = torch::sparse_compressed_tensor(
                    empty_compressed,
                    empty_plain,
                    empty_values,
                    c10::IntArrayRef(size_vec),
                    torch::TensorOptions().layout(torch::kSparse).dtype(values.dtype())
                );
            } catch (const c10::Error& e) {
                // PyTorch specific errors are expected and okay
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