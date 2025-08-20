#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create meaningful input
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to use with ShortStorage
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a ShortStorage from the tensor
        try {
            // Convert tensor to short type if needed
            if (tensor.dtype() != torch::kInt16) {
                tensor = tensor.to(torch::kInt16);
            }
            
            // Create ShortStorage from tensor
            auto storage = c10::Storage(c10::make_intrusive<c10::StorageImpl>(
                c10::StorageImpl::use_byte_size_t(),
                tensor.numel() * sizeof(int16_t),
                c10::GetDefaultCPUAllocator(),
                true));
            
            // Fill storage with tensor data
            auto tensor_contiguous = tensor.reshape(-1).contiguous();
            auto tensor_accessor = tensor_contiguous.accessor<int16_t, 1>();
            int16_t* storage_data = static_cast<int16_t*>(storage.data());
            for (int64_t i = 0; i < tensor.numel() && i < storage.nbytes() / sizeof(int16_t); i++) {
                storage_data[i] = tensor_accessor[i];
            }
            
            // Test storage properties
            int64_t size = storage.nbytes() / sizeof(int16_t);
            
            // Test data access
            if (storage.data() && size > 0) {
                int16_t first_element = storage_data[0];
                
                // Test element assignment if size allows
                if (size > 1) {
                    storage_data[1] = 42;
                    int16_t modified_element = storage_data[1];
                }
            }
            
            // Test copy constructor
            auto storage_copy = storage;
            
            // Test move constructor
            auto storage_moved = std::move(storage_copy);
            
            // Test resize
            if (offset + sizeof(int64_t) <= Size) {
                int64_t new_size;
                std::memcpy(&new_size, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // Ensure new_size is reasonable
                new_size = std::abs(new_size) % 1000;
                
                storage_moved.resize(new_size * sizeof(int16_t));
            }
            
            // Test fill
            if (offset < Size && storage_moved.data()) {
                int16_t fill_value = static_cast<int16_t>(Data[offset++]);
                int16_t* moved_data = static_cast<int16_t*>(storage_moved.data());
                int64_t moved_size = storage_moved.nbytes() / sizeof(int16_t);
                for (int64_t i = 0; i < moved_size; i++) {
                    moved_data[i] = fill_value;
                }
            }
            
            // Create a tensor from the storage
            int64_t storage_size = storage_moved.nbytes() / sizeof(int16_t);
            auto tensor_from_storage = torch::empty({storage_size}, torch::kInt16);
            if (storage_moved.data() && storage_size > 0) {
                auto accessor = tensor_from_storage.accessor<int16_t, 1>();
                int16_t* moved_data = static_cast<int16_t*>(storage_moved.data());
                for (int64_t i = 0; i < storage_size; i++) {
                    accessor[i] = moved_data[i];
                }
            }
        }
        catch (const c10::Error &e) {
            // PyTorch specific errors are expected and okay
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}