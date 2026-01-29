#include "fuzzer_utils.h" // General fuzzing utilities
#include <cstring>        // For memcpy
#include <iostream>       // For cerr
#include <typeinfo>       // For type checking

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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test storage operations on the tensor's storage
        // torch.is_storage equivalent: check if object is a storage
        // In C++, we check storage validity via data pointer or nbytes
        
        if (tensor.defined() && tensor.numel() > 0) {
            auto storage = tensor.storage();
            
            // Check if storage is valid (C++ equivalent of is_storage check)
            // Storage is valid if it has allocated bytes
            bool storage_valid = (storage.nbytes() > 0);
            (void)storage_valid;
            
            // Test storage properties when valid
            if (storage.nbytes() > 0) {
                auto storage_nbytes = storage.nbytes();
                auto storage_device = storage.device();
                // Get dtype from tensor since storage doesn't expose it directly
                auto tensor_dtype = tensor.dtype();
                (void)storage_nbytes;
                (void)storage_device;
                (void)tensor_dtype;
                
                // Test data pointer access
                const void *storage_data = storage.data();
                (void)storage_data;
            }
        }
        
        // Test with different tensor types and their storages
        if (offset < Size) {
            uint8_t type_selector = Data[offset++] % 4;
            
            torch::Tensor typed_tensor;
            try {
                switch (type_selector) {
                    case 0:
                        typed_tensor = torch::zeros({2, 2}, torch::kFloat);
                        break;
                    case 1:
                        typed_tensor = torch::ones({3}, torch::kDouble);
                        break;
                    case 2:
                        typed_tensor = torch::randint(0, 100, {4}, torch::kInt);
                        break;
                    case 3:
                        typed_tensor = torch::zeros({2}, torch::kLong);
                        break;
                }
                
                if (typed_tensor.defined()) {
                    auto typed_storage = typed_tensor.storage();
                    bool typed_storage_valid = (typed_storage.nbytes() > 0);
                    (void)typed_storage_valid;
                    
                    // Verify storage shares data with tensor
                    if (typed_storage.nbytes() > 0) {
                        // Access element size from tensor's dtype
                        auto elem_size = typed_tensor.element_size();
                        (void)elem_size;
                    }
                }
            }
            catch (...) {
                // Silently handle expected failures
            }
        }
        
        // Test storage of a cloned tensor
        if (tensor.defined() && tensor.numel() > 0) {
            try {
                auto cloned = tensor.clone();
                auto cloned_storage = cloned.storage();
                
                // Cloned tensor should have its own storage
                bool cloned_storage_valid = (cloned_storage.nbytes() > 0);
                (void)cloned_storage_valid;
                
                // Storages should be different objects
                if (cloned_storage.nbytes() > 0 && tensor.storage().nbytes() > 0) {
                    bool same_storage = (cloned_storage.data() == tensor.storage().data());
                    (void)same_storage; // Should be false for clone
                }
            }
            catch (...) {
                // Silently handle expected failures
            }
        }
        
        // Test view tensor storage (should share storage)
        if (tensor.defined() && tensor.numel() > 1) {
            try {
                auto view = tensor.view({-1});
                auto view_storage = view.storage();
                
                if (view_storage.nbytes() > 0 && tensor.storage().nbytes() > 0) {
                    // View should share storage with original
                    bool shares_storage = (view_storage.data() == tensor.storage().data());
                    (void)shares_storage; // Should be true for view
                }
            }
            catch (...) {
                // Silently handle expected failures (e.g., non-contiguous tensors)
            }
        }
        
        // Test empty tensor storage
        {
            auto empty_tensor = torch::empty({0});
            if (empty_tensor.defined()) {
                auto empty_storage = empty_tensor.storage();
                // Empty tensor may have 0 bytes storage
                bool empty_storage_valid = (empty_storage.nbytes() >= 0);
                (void)empty_storage_valid;
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