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
        if (Size < 10) {
            return 0;
        }
        
        // Create values tensor
        torch::Tensor values = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse ccol_indices tensor (note: BSC uses ccol_indices, not crow_indices)
        torch::Tensor ccol_indices = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse row_indices tensor
        torch::Tensor row_indices = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse sparse dimensions
        std::vector<int64_t> sparse_dims;
        if (offset + 2 <= Size) {
            uint8_t sparse_dim_count = Data[offset++] % 3 + 1; // 1-3 sparse dimensions
            for (uint8_t i = 0; i < sparse_dim_count && offset < Size; i++) {
                int64_t dim = static_cast<int64_t>(Data[offset++]) % 10 + 1; // 1-10 dimension size
                sparse_dims.push_back(dim);
            }
        } else {
            // Default sparse dimensions if not enough data
            sparse_dims = {2, 3};
        }
        
        // Create sparse BSC tensor with size parameter
        try {
            torch::Tensor sparse_bsc = torch::sparse_bsc_tensor(
                ccol_indices,
                row_indices,
                values,
                sparse_dims
            );
            
            // Test some operations on the sparse tensor
            if (sparse_bsc.defined()) {
                auto indices = sparse_bsc.indices();
                auto values_out = sparse_bsc.values();
                auto dense = sparse_bsc.to_dense();
            }
        } catch (const c10::Error& e) {
            // Expected exceptions from PyTorch operations are fine
        }
        
        // Try with TensorOptions
        if (offset < Size) {
            try {
                auto options = torch::TensorOptions().dtype(torch::kFloat32);
                torch::Tensor sparse_bsc_with_options = torch::sparse_bsc_tensor(
                    ccol_indices,
                    row_indices,
                    values,
                    sparse_dims,
                    options
                );
            } catch (const c10::Error& e) {
                // Expected exceptions from PyTorch operations are fine
            }
        }
        
        // Try with different dtype in options
        if (offset < Size) {
            try {
                auto dtype_selector = Data[offset++];
                auto dtype = fuzzer_utils::parseDataType(dtype_selector);
                auto options = torch::TensorOptions().dtype(dtype);
                torch::Tensor sparse_bsc_with_dtype = torch::sparse_bsc_tensor(
                    ccol_indices,
                    row_indices,
                    values,
                    sparse_dims,
                    options
                );
            } catch (const c10::Error& e) {
                // Expected exceptions from PyTorch operations are fine
            }
        }
        
        // Try with device in options
        if (offset < Size) {
            try {
                auto device = torch::kCPU;
                auto options = torch::TensorOptions().device(device);
                torch::Tensor sparse_bsc_with_device = torch::sparse_bsc_tensor(
                    ccol_indices,
                    row_indices,
                    values,
                    sparse_dims,
                    options
                );
            } catch (const c10::Error& e) {
                // Expected exceptions from PyTorch operations are fine
            }
        }
        
        // Try with all options combined
        if (offset < Size) {
            try {
                auto dtype_selector = Data[offset++];
                auto dtype = fuzzer_utils::parseDataType(dtype_selector);
                auto device = torch::kCPU;
                auto options = torch::TensorOptions().dtype(dtype).device(device);
                torch::Tensor sparse_bsc_full = torch::sparse_bsc_tensor(
                    ccol_indices,
                    row_indices,
                    values,
                    sparse_dims,
                    options
                );
            } catch (const c10::Error& e) {
                // Expected exceptions from PyTorch operations are fine
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