#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>
#include <vector>

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
        (void)storage_size;
        (void)storage_data;
        
        // Test storage type through tensor
        auto storage_dtype = tensor.dtype();
        
        // Test storage element access through tensor if it has elements
        if (tensor.numel() > 0) {
            // Access first element through tensor
            auto first_element = tensor.flatten()[0];
            (void)first_element;
            
            // Access last element through tensor
            auto last_element = tensor.flatten()[tensor.numel() - 1];
            (void)last_element;
            
            // Test storage copy through tensor
            auto tensor_copy = tensor.clone();
            auto storage_copy = tensor_copy.storage();
            (void)storage_copy;
            
            // Test tensor resize if there's more data available
            if (offset < Size) {
                uint8_t resize_value = Data[offset++];
                size_t new_size = resize_value % 32 + 1;
                try {
                    tensor.resize_({static_cast<int64_t>(new_size)});
                } catch (...) {
                    // Resize may fail for certain tensor types, ignore
                }
            }
            
            // Test tensor fill if there's more data available
            if (offset < Size) {
                try {
                    if (storage_dtype == torch::kFloat32 || storage_dtype == torch::kFloat64) {
                        float fill_value = static_cast<float>(Data[offset++]) / 255.0f;
                        tensor.fill_(fill_value);
                    } else if (storage_dtype == torch::kInt64 || storage_dtype == torch::kInt32) {
                        int64_t fill_value = static_cast<int64_t>(Data[offset++]);
                        tensor.fill_(fill_value);
                    } else if (storage_dtype == torch::kBool) {
                        bool fill_value = Data[offset++] % 2 == 0;
                        tensor.fill_(fill_value);
                    } else {
                        offset++;
                    }
                } catch (...) {
                    // Fill may fail for certain tensor types
                }
            }
        }
        
        // Test storage creation through tensor
        if (offset + 4 <= Size) {
            std::vector<float> data_vec(4);
            for (int i = 0; i < 4 && offset < Size; i++) {
                data_vec[i] = static_cast<float>(Data[offset++]);
            }
            
            auto new_tensor = torch::from_blob(
                data_vec.data(),
                {static_cast<int64_t>(data_vec.size())},
                torch::kFloat32
            ).clone();
            
            auto new_storage = new_tensor.storage();
            
            size_t new_storage_size = new_storage.nbytes();
            const void* new_storage_data = new_storage.data();
            (void)new_storage_size;
            (void)new_storage_data;
        }
        
        // Test tensor slicing and its effect on storage
        if (tensor.numel() > 1) {
            size_t offset_val = 0;
            if (offset < Size) {
                offset_val = Data[offset++] % static_cast<size_t>(tensor.numel());
            }
            
            size_t length = 1;
            if (offset < Size && static_cast<size_t>(tensor.numel()) > offset_val + 1) {
                length = Data[offset++] % (static_cast<size_t>(tensor.numel()) - offset_val);
                if (length == 0) length = 1;
            }
            
            try {
                auto tensor_slice = tensor.flatten().slice(0, 
                    static_cast<int64_t>(offset_val), 
                    static_cast<int64_t>(offset_val + length));
                auto storage_view = tensor_slice.storage();
                
                size_t view_size = storage_view.nbytes();
                const void* view_data = storage_view.data();
                (void)view_size;
                (void)view_data;
            } catch (...) {
                // Slice may fail with invalid indices
            }
        }
        
        // Test tensor reshape and storage sharing
        if (tensor.numel() > 0 && offset < Size) {
            uint8_t dim_count = Data[offset++] % 4 + 1;
            
            std::vector<int64_t> sizes;
            int64_t total_size = 1;
            for (int i = 0; i < dim_count && offset < Size; i++) {
                int64_t dim_size = Data[offset++] % 8 + 1;
                sizes.push_back(dim_size);
                total_size *= dim_size;
            }
            
            if (!sizes.empty() && total_size <= tensor.numel()) {
                try {
                    auto tensor_reshaped = tensor.reshape(sizes);
                    auto reshaped_storage = tensor_reshaped.storage();
                    
                    // Verify storage is shared
                    bool same_storage = (storage.data() == reshaped_storage.data());
                    (void)same_storage;
                } catch (...) {
                    // Reshape may fail
                }
            }
        }
        
        // Test storage offset through tensor views
        if (tensor.dim() > 0 && tensor.size(0) > 1) {
            try {
                auto view = tensor.narrow(0, 1, 1);
                auto view_storage = view.storage();
                
                // View should share storage but have different offset
                int64_t view_offset = view.storage_offset();
                (void)view_offset;
            } catch (...) {
                // Narrow may fail
            }
        }
        
        // Test is_alias_of for storage sharing detection
        if (tensor.numel() > 0) {
            auto alias = tensor.view({-1});
            bool is_alias = tensor.is_alias_of(alias);
            (void)is_alias;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}