#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        
        // Need enough bytes for parameters
        if (Size < 16) {
            return 0;
        }
        
        // Parse block dimensions (block_size for BSC format)
        int64_t block_rows = static_cast<int64_t>(Data[offset++] % 4) + 1;  // 1-4
        int64_t block_cols = static_cast<int64_t>(Data[offset++] % 4) + 1;  // 1-4
        
        // Parse number of block rows and columns in the sparse matrix
        int64_t num_block_rows = static_cast<int64_t>(Data[offset++] % 4) + 1;  // 1-4
        int64_t num_block_cols = static_cast<int64_t>(Data[offset++] % 4) + 1;  // 1-4
        
        // Parse number of non-zero blocks (limited to valid range)
        int64_t max_nnz = num_block_rows * num_block_cols;
        int64_t nnz_blocks = static_cast<int64_t>(Data[offset++] % std::max(max_nnz, (int64_t)1)) + 1;
        if (nnz_blocks > max_nnz) {
            nnz_blocks = max_nnz;
        }
        
        // Full tensor size
        std::vector<int64_t> tensor_size = {
            num_block_rows * block_rows,
            num_block_cols * block_cols
        };
        
        // Create ccol_indices: size (num_block_cols + 1,), values in range [0, nnz_blocks]
        std::vector<int64_t> ccol_data(num_block_cols + 1);
        ccol_data[0] = 0;
        for (int64_t i = 1; i <= num_block_cols; i++) {
            int64_t increment = 0;
            if (offset < Size) {
                increment = static_cast<int64_t>(Data[offset++] % (nnz_blocks / num_block_cols + 2));
            }
            ccol_data[i] = std::min(ccol_data[i-1] + increment, nnz_blocks);
        }
        ccol_data[num_block_cols] = nnz_blocks;  // Last element must equal nnz_blocks
        
        torch::Tensor ccol_indices = torch::tensor(ccol_data, torch::kInt64);
        
        // Create row_indices: size (nnz_blocks,), values in range [0, num_block_rows)
        std::vector<int64_t> row_data(nnz_blocks);
        for (int64_t i = 0; i < nnz_blocks; i++) {
            if (offset < Size) {
                row_data[i] = static_cast<int64_t>(Data[offset++] % num_block_rows);
            } else {
                row_data[i] = i % num_block_rows;
            }
        }
        torch::Tensor row_indices = torch::tensor(row_data, torch::kInt64);
        
        // Create values tensor: shape (nnz_blocks, block_rows, block_cols)
        torch::Tensor values = fuzzer_utils::createTensor(Data, Size, offset);
        // Reshape to proper block dimensions
        try {
            values = values.flatten().slice(0, 0, nnz_blocks * block_rows * block_cols);
            if (values.numel() < nnz_blocks * block_rows * block_cols) {
                // Pad if necessary
                values = torch::cat({values, torch::zeros(nnz_blocks * block_rows * block_cols - values.numel())});
            }
            values = values.reshape({nnz_blocks, block_rows, block_cols});
        } catch (...) {
            // If reshaping fails, create a proper values tensor
            values = torch::randn({nnz_blocks, block_rows, block_cols});
        }
        
        // Create sparse BSC tensor (requires 5 arguments: ccol_indices, row_indices, values, size, options)
        try {
            auto options = torch::TensorOptions().dtype(values.dtype());
            torch::Tensor sparse_bsc = torch::sparse_bsc_tensor(
                ccol_indices,
                row_indices,
                values,
                tensor_size,
                options
            );
            
            // Test operations on the sparse tensor
            if (sparse_bsc.defined()) {
                auto ccol_out = sparse_bsc.ccol_indices();
                auto row_out = sparse_bsc.row_indices();
                auto values_out = sparse_bsc.values();
                auto dense = sparse_bsc.to_dense();
                
                // Test properties
                auto nnz = sparse_bsc._nnz();
                auto is_sparse = sparse_bsc.is_sparse();
            }
        } catch (const c10::Error& e) {
            // Expected exceptions from invalid sparse tensor construction
        }
        
        // Try with TensorOptions - float32
        try {
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            torch::Tensor values_f32 = values.to(torch::kFloat32);
            torch::Tensor sparse_bsc_f32 = torch::sparse_bsc_tensor(
                ccol_indices,
                row_indices,
                values_f32,
                tensor_size,
                options
            );
        } catch (const c10::Error& e) {
            // Expected exceptions
        }
        
        // Try with different dtype
        if (offset < Size) {
            try {
                auto dtype_selector = Data[offset++];
                auto dtype = fuzzer_utils::parseDataType(dtype_selector);
                auto options = torch::TensorOptions().dtype(dtype);
                torch::Tensor values_typed = values.to(dtype);
                torch::Tensor sparse_bsc_typed = torch::sparse_bsc_tensor(
                    ccol_indices,
                    row_indices,
                    values_typed,
                    tensor_size,
                    options
                );
            } catch (const c10::Error& e) {
                // Expected exceptions
            }
        }
        
        // Try with float64
        try {
            auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
            torch::Tensor values_f64 = values.to(torch::kFloat64);
            torch::Tensor sparse_bsc_f64 = torch::sparse_bsc_tensor(
                ccol_indices,
                row_indices,
                values_f64,
                tensor_size,
                options
            );
            
            if (sparse_bsc_f64.defined()) {
                // Test conversion
                auto dense_f64 = sparse_bsc_f64.to_dense();
            }
        } catch (const c10::Error& e) {
            // Expected exceptions
        }
        
        // Try with complex type
        try {
            auto options = torch::TensorOptions().dtype(torch::kComplexFloat);
            torch::Tensor values_complex = values.to(torch::kComplexFloat);
            torch::Tensor sparse_bsc_complex = torch::sparse_bsc_tensor(
                ccol_indices,
                row_indices,
                values_complex,
                tensor_size,
                options
            );
        } catch (const c10::Error& e) {
            // Expected exceptions
        }
        
        // Try without specifying size (3-argument version)
        try {
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            torch::Tensor values_f32 = values.to(torch::kFloat32);
            torch::Tensor sparse_bsc_no_size = torch::sparse_bsc_tensor(
                ccol_indices,
                row_indices,
                values_f32,
                options
            );
        } catch (const c10::Error& e) {
            // Expected exceptions
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}