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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get the storage from the tensor
        auto storage = tensor.storage();
        
        // Test various storage properties and methods
        size_t storage_size = storage.nbytes();
        const void* storage_data = storage.data();
        
        // Test storage type through tensor
        auto storage_dtype = tensor.dtype();
        
        // Test storage element access through tensor if it has elements
        if (tensor.numel() > 0) {
            // Access first element through tensor
            auto first_element = tensor.flatten()[0];
            
            // Access last element through tensor
            auto last_element = tensor.flatten()[tensor.numel() - 1];
            
            // Test storage copy through tensor
            auto tensor_copy = tensor.clone();
            auto storage_copy = tensor_copy.storage();
            
            // Test tensor resize if there's more data available
            if (offset < Size) {
                uint8_t resize_value = Data[offset++];
                size_t new_size = resize_value % 32 + 1; // Keep size reasonable and non-zero
                tensor.resize_({static_cast<int64_t>(new_size)});
            }
            
            // Test tensor fill if there's more data available
            if (offset < Size) {
                // Fill with a scalar value
                if (storage_dtype == torch::kFloat) {
                    float fill_value = static_cast<float>(Data[offset++]) / 255.0f;
                    tensor.fill_(fill_value);
                } else if (storage_dtype == torch::kInt64) {
                    int64_t fill_value = static_cast<int64_t>(Data[offset++]);
                    tensor.fill_(fill_value);
                } else if (storage_dtype == torch::kBool) {
                    bool fill_value = Data[offset++] % 2 == 0;
                    tensor.fill_(fill_value);
                }
            }
        }
        
        // Test storage creation through tensor
        if (offset + 4 <= Size) {
            // Create a small array of data
            std::vector<float> data_vec(4);
            for (int i = 0; i < 4 && offset < Size; i++) {
                data_vec[i] = static_cast<float>(Data[offset++]);
            }
            
            // Create tensor from data
            auto new_tensor = torch::from_blob(
                data_vec.data(),
                {static_cast<int64_t>(data_vec.size())},
                torch::kFloat
            ).clone();
            
            // Get storage from new tensor
            auto new_storage = new_tensor.storage();
            
            // Test properties of the new storage
            size_t new_storage_size = new_storage.nbytes();
            const void* new_storage_data = new_storage.data();
        }
        
        // Test tensor slicing instead of storage slicing
        if (tensor.numel() > 1) {
            size_t offset_val = 0;
            if (offset < Size) {
                offset_val = Data[offset++] % tensor.numel();
            }
            
            size_t length = 1;
            if (offset < Size && tensor.numel() > offset_val + 1) {
                length = Data[offset++] % (tensor.numel() - offset_val);
                if (length == 0) length = 1;
            }
            
            auto tensor_slice = tensor.flatten().slice(0, offset_val, offset_val + length);
            auto storage_view = tensor_slice.storage();
            
            // Test properties of the storage view
            size_t view_size = storage_view.nbytes();
            const void* view_data = storage_view.data();
        }
        
        // Test tensor from storage concept
        if (tensor.numel() > 0) {
            // Create a tensor with different shape from same storage
            if (offset < Size) {
                uint8_t dim_count = Data[offset++] % 4 + 1;
                
                std::vector<int64_t> sizes;
                int64_t total_size = 1;
                for (int i = 0; i < dim_count && offset < Size; i++) {
                    int64_t dim_size = Data[offset++] % 8 + 1;
                    sizes.push_back(dim_size);
                    total_size *= dim_size;
                }
                
                // Ensure total size doesn't exceed tensor size
                if (total_size <= tensor.numel()) {
                    auto tensor_reshaped = tensor.reshape(sizes);
                    auto reshaped_storage = tensor_reshaped.storage();
                }
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
