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
        
        // Parse indices tensor (should be integer type)
        torch::Tensor crow_indices = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor col_indices = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse sparse dimensions
        std::vector<int64_t> sparse_dims;
        if (offset + 2 < Size) {
            uint8_t rank = Data[offset++] % 3 + 2; // 2-4 dimensions
            for (uint8_t i = 0; i < rank && offset < Size; i++) {
                int64_t dim = static_cast<int64_t>(Data[offset++]) + 1; // Ensure positive dimension
                sparse_dims.push_back(dim);
            }
        } else {
            // Default dimensions if not enough data
            sparse_dims = {3, 4};
        }
        
        // Try different combinations of parameters
        try {
            // Basic sparse_csc_tensor creation
            auto sparse_tensor = torch::sparse_csc_tensor(
                crow_indices,
                col_indices,
                values,
                sparse_dims,
                torch::TensorOptions()
            );
            
            // Test some operations on the sparse tensor
            if (sparse_tensor.defined()) {
                auto dense = sparse_tensor.to_dense();
                auto indices = sparse_tensor.indices();
                auto values_extracted = sparse_tensor.values();
            }
        } catch (const c10::Error& e) {
            // Expected exceptions from PyTorch operations
        }
        
        // Try with different dtype
        if (offset < Size) {
            torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset++]);
            try {
                auto sparse_tensor_with_dtype = torch::sparse_csc_tensor(
                    crow_indices,
                    col_indices,
                    values,
                    sparse_dims,
                    torch::TensorOptions().dtype(dtype)
                );
            } catch (const c10::Error& e) {
                // Expected exceptions from PyTorch operations
            }
        }
        
        // Try with layout parameter
        if (offset < Size) {
            bool use_sparse_csc = Data[offset++] % 2;
            auto layout = use_sparse_csc ? torch::kSparseCsc : torch::kSparseCsr;
            try {
                auto sparse_tensor_with_layout = torch::sparse_csc_tensor(
                    crow_indices,
                    col_indices,
                    values,
                    sparse_dims,
                    torch::TensorOptions().dtype(values.scalar_type()).layout(layout)
                );
            } catch (const c10::Error& e) {
                // Expected exceptions from PyTorch operations
            }
        }
        
        // Try with device parameter
        if (offset < Size) {
            try {
                auto sparse_tensor_with_device = torch::sparse_csc_tensor(
                    crow_indices,
                    col_indices,
                    values,
                    sparse_dims,
                    torch::TensorOptions().dtype(values.scalar_type()).layout(torch::kSparseCsc).device(torch::kCPU)
                );
            } catch (const c10::Error& e) {
                // Expected exceptions from PyTorch operations
            }
        }
        
        // Try with requires_grad parameter
        if (offset < Size) {
            bool requires_grad = Data[offset++] % 2;
            try {
                auto sparse_tensor_with_grad = torch::sparse_csc_tensor(
                    crow_indices,
                    col_indices,
                    values,
                    sparse_dims,
                    torch::TensorOptions().dtype(values.scalar_type()).layout(torch::kSparseCsc).device(torch::kCPU).requires_grad(requires_grad)
                );
            } catch (const c10::Error& e) {
                // Expected exceptions from PyTorch operations
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
