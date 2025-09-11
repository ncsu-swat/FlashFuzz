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
        
        // Parse blocksize
        std::vector<int64_t> blocksize;
        if (offset + 2 <= Size) {
            uint8_t blocksize_dim1 = Data[offset++] % 8 + 1;
            uint8_t blocksize_dim2 = Data[offset++] % 8 + 1;
            blocksize = {static_cast<int64_t>(blocksize_dim1), static_cast<int64_t>(blocksize_dim2)};
        } else {
            blocksize = {2, 2}; // Default blocksize
        }
        
        // Parse size (optional)
        std::vector<int64_t> size;
        if (offset + 1 < Size) {
            uint8_t use_size = Data[offset++] % 2;
            if (use_size && offset + 2 <= Size) {
                uint8_t size_dim1 = Data[offset++] % 32 + 1;
                uint8_t size_dim2 = Data[offset++] % 32 + 1;
                size = {static_cast<int64_t>(size_dim1), static_cast<int64_t>(size_dim2)};
            }
        }
        
        // Parse layout (optional)
        torch::Layout layout = torch::kStrided;
        if (offset < Size) {
            uint8_t layout_byte = Data[offset++];
            if (layout_byte % 2 == 1) {
                layout = torch::kSparse;
            }
        }
        
        // Parse device (CPU only for fuzzing)
        torch::Device device = torch::kCPU;
        
        // Parse requires_grad (optional)
        bool requires_grad = false;
        if (offset < Size) {
            requires_grad = (Data[offset++] % 2 == 1);
        }
        
        // Create sparse_bsr_tensor with different parameter combinations
        try {
            torch::Tensor result;
            
            // Try different combinations of parameters
            if (offset < Size) {
                uint8_t param_choice = Data[offset++] % 4;
                
                switch (param_choice) {
                    case 0:
                        // Basic version with TensorOptions (required)
                        {
                            auto options = torch::TensorOptions()
                                .dtype(values.dtype())
                                .layout(layout)
                                .device(device)
                                .requires_grad(requires_grad);
                            result = torch::sparse_bsr_tensor(crow_indices, col_indices, values, options);
                        }
                        break;
                    case 1:
                        // With size parameter
                        {
                            auto options = torch::TensorOptions()
                                .dtype(values.dtype())
                                .layout(layout)
                                .device(device)
                                .requires_grad(requires_grad);
                            if (!size.empty()) {
                                result = torch::sparse_bsr_tensor(crow_indices, col_indices, values, size, options);
                            } else {
                                result = torch::sparse_bsr_tensor(crow_indices, col_indices, values, options);
                            }
                        }
                        break;
                    case 2:
                        // With TensorOptions
                        {
                            auto options = torch::TensorOptions()
                                .dtype(values.dtype())
                                .layout(layout)
                                .device(device)
                                .requires_grad(requires_grad);
                            
                            if (!size.empty()) {
                                result = torch::sparse_bsr_tensor(crow_indices, col_indices, values, size, options);
                            } else {
                                result = torch::sparse_bsr_tensor(crow_indices, col_indices, values, options);
                            }
                        }
                        break;
                    case 3:
                        // With all parameters
                        {
                            auto options = torch::TensorOptions()
                                .dtype(values.dtype())
                                .layout(layout)
                                .device(device)
                                .requires_grad(requires_grad);
                            
                            if (!size.empty()) {
                                result = torch::sparse_bsr_tensor(crow_indices, col_indices, values, size, options);
                            } else {
                                result = torch::sparse_bsr_tensor(crow_indices, col_indices, values, options);
                            }
                        }
                        break;
                }
            } else {
                // Default case if we don't have enough data
                auto options = torch::TensorOptions()
                    .dtype(values.dtype())
                    .layout(layout)
                    .device(device)
                    .requires_grad(requires_grad);
                result = torch::sparse_bsr_tensor(crow_indices, col_indices, values, options);
            }
            
            // Perform some operations on the result to ensure it's valid
            auto indices = result._indices();
            auto values_result = result._values();
            auto sparse_dim = result.sparse_dim();
            auto dense_dim = result.dense_dim();
            auto is_coalesced = result.is_coalesced();
            
            // Try to coalesce if not already coalesced
            if (!is_coalesced) {
                try {
                    auto coalesced = result.coalesce();
                } catch (...) {
                    // Ignore coalescing errors
                }
            }
            
            // Try to convert to dense
            try {
                auto dense = result.to_dense();
            } catch (...) {
                // Ignore conversion errors
            }
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors but don't terminate fuzzing
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
