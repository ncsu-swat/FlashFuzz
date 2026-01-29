#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor to test layout properties
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test torch.layout functionality
        auto layout = tensor.layout();
        
        // Test layout properties and operations
        bool is_strided = layout == torch::kStrided;
        bool is_sparse = layout == torch::kSparse;
        bool is_sparse_csr = layout == torch::kSparseCsr;
        bool is_sparse_csc = layout == torch::kSparseCsc;
        bool is_sparse_bsr = layout == torch::kSparseBsr;
        bool is_sparse_bsc = layout == torch::kSparseBsc;
        
        // Prevent optimization
        volatile bool v1 = is_strided;
        volatile bool v2 = is_sparse;
        volatile bool v3 = is_sparse_csr;
        volatile bool v4 = is_sparse_csc;
        volatile bool v5 = is_sparse_bsr;
        volatile bool v6 = is_sparse_bsc;
        (void)v1; (void)v2; (void)v3; (void)v4; (void)v5; (void)v6;
        
        // Test layout comparison
        bool equals_strided = (layout == torch::kStrided);
        bool not_equals_sparse = (layout != torch::kSparse);
        volatile bool v7 = equals_strided;
        volatile bool v8 = not_equals_sparse;
        (void)v7; (void)v8;
        
        // Test layout conversion to string
        std::string layout_str = c10::toString(layout);
        
        // Create tensors with different layouts if possible
        if (offset + 1 < Size) {
            try {
                // Try to create a sparse tensor
                auto indices = torch::zeros({2, 1}, torch::kLong);
                auto values = torch::ones({1});
                auto sparse_tensor = torch::sparse_coo_tensor(indices, values, {3, 3});
                auto sparse_layout = sparse_tensor.layout();
                
                // Test sparse layout properties
                bool sparse_is_sparse = sparse_layout == torch::kSparse;
                bool sparse_not_strided = sparse_layout != torch::kStrided;
                volatile bool v9 = sparse_is_sparse;
                volatile bool v10 = sparse_not_strided;
                (void)v9; (void)v10;
            } catch (...) {
                // Sparse tensor creation might fail, that's okay
            }
        }
        
        // Test layout with different tensor operations
        try {
            // Clone preserves layout
            auto cloned = tensor.clone();
            bool clone_same_layout = (cloned.layout() == tensor.layout());
            volatile bool v11 = clone_same_layout;
            (void)v11;
            
            // Reshape preserves layout
            if (!tensor.sizes().empty() && tensor.numel() > 0) {
                auto new_shape = std::vector<int64_t>{tensor.numel()};
                auto reshaped = tensor.reshape(new_shape);
                bool reshape_same_layout = (reshaped.layout() == tensor.layout());
                volatile bool v12 = reshape_same_layout;
                (void)v12;
            }
            
            // Test to_dense/to_sparse if applicable
            if (tensor.layout() == torch::kStrided && tensor.dim() > 0) {
                try {
                    auto sparse_version = tensor.to_sparse();
                    bool sparse_has_sparse_layout = (sparse_version.layout() == torch::kSparse);
                    volatile bool v13 = sparse_has_sparse_layout;
                    (void)v13;
                } catch (...) {
                    // to_sparse might fail for some tensors
                }
            }
        } catch (...) {
            // Some operations might fail depending on tensor properties
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}