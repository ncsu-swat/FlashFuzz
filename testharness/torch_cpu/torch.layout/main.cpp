#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
        
        // Test layout comparison
        bool equals_strided = (layout == torch::kStrided);
        bool not_equals_sparse = (layout != torch::kSparse);
        
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
            } catch (const std::exception& e) {
                // Sparse tensor creation might fail, that's okay
            }
        }
        
        // Test layout with different tensor operations
        try {
            // Clone preserves layout
            auto cloned = tensor.clone();
            bool clone_same_layout = (cloned.layout() == tensor.layout());
            
            // Reshape preserves layout
            if (!tensor.sizes().empty() && tensor.numel() > 0) {
                auto new_shape = std::vector<int64_t>{tensor.numel()};
                auto reshaped = tensor.reshape(new_shape);
                bool reshape_same_layout = (reshaped.layout() == tensor.layout());
            }
            
            // Test to_dense/to_sparse if applicable
            if (tensor.layout() == torch::kStrided && tensor.dim() > 0) {
                try {
                    auto sparse_version = tensor.to_sparse();
                    bool sparse_has_sparse_layout = (sparse_version.layout() == torch::kSparse);
                } catch (const std::exception& e) {
                    // to_sparse might fail for some tensors
                }
            }
        } catch (const std::exception& e) {
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