#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor with complex float values
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert tensor to complex float if it's not already
        if (tensor.dtype() != torch::kComplexFloat) {
            tensor = tensor.to(torch::kComplexFloat);
        }
        
        // Get the storage from the tensor
        auto storage = tensor.storage();
        
        // Test various operations on the storage
        size_t storage_size = storage.nbytes() / sizeof(c10::complex<float>);
        
        // Access elements if storage is not empty
        if (storage_size > 0) {
            // Access data pointer
            auto data_ptr = static_cast<c10::complex<float>*>(storage.data());
            
            // Access first element
            auto first_element = data_ptr[0];
            
            // Access last element
            auto last_element = data_ptr[storage_size - 1];
            
            // Access random element if size > 1
            if (storage_size > 1 && offset < Size) {
                size_t idx = Data[offset++] % storage_size;
                auto random_element = data_ptr[idx];
            }
        }
        
        // Test data() method
        auto data_ptr = storage.data();
        
        // Test nbytes() method
        auto nbytes = storage.nbytes();
        
        // Test device() method
        auto device = storage.device();
        
        // Test is_pinned() method
        auto is_pinned = storage.is_pinned();
        
        // Test copy operations if we have enough data
        if (offset + 1 < Size) {
            // Create a new tensor with complex float type
            auto new_tensor = torch::zeros({static_cast<int64_t>(storage_size)}, torch::kComplexFloat);
            auto new_storage = new_tensor.storage();
            
            // Copy data manually
            if (storage_size > 0) {
                std::memcpy(new_storage.data(), storage.data(), storage.nbytes());
            }
            
            // Create a partial copy if storage is not empty
            if (storage_size > 0) {
                size_t partial_size = std::max<size_t>(1, storage_size / 2);
                auto partial_tensor = torch::zeros({static_cast<int64_t>(partial_size)}, torch::kComplexFloat);
                auto partial_storage = partial_tensor.storage();
                std::memcpy(partial_storage.data(), storage.data(), partial_size * sizeof(c10::complex<float>));
            }
        }
        
        // Test resize operations by creating a new tensor
        if (offset < Size) {
            size_t new_size = Data[offset++] % 100;
            auto resized_tensor = torch::zeros({static_cast<int64_t>(new_size)}, torch::kComplexFloat);
        }
        
        // Test fill operation
        if (offset + sizeof(c10::complex<float>) <= Size && storage_size > 0) {
            c10::complex<float> fill_value;
            std::memcpy(&fill_value, Data + offset, sizeof(c10::complex<float>));
            offset += sizeof(c10::complex<float>);
            
            // Fill the tensor data
            auto data_ptr = static_cast<c10::complex<float>*>(storage.data());
            for (size_t i = 0; i < storage_size; ++i) {
                data_ptr[i] = fill_value;
            }
        }
        
        // Test creating a tensor from storage
        if (storage_size > 0) {
            auto tensor_from_storage = torch::from_blob(storage.data(), {static_cast<int64_t>(storage_size)}, 
                                                       torch::TensorOptions().dtype(torch::kComplexFloat));
        }
        
        // Test creating a storage with specific size
        if (offset < Size) {
            size_t custom_size = Data[offset++] % 100;
            auto custom_tensor = torch::zeros({static_cast<int64_t>(custom_size)}, torch::kComplexFloat);
            auto custom_storage = custom_tensor.storage();
        }
        
        // Test creating a storage from data pointer
        if (storage_size > 0) {
            auto storage_from_data = torch::from_blob(storage.data(), {static_cast<int64_t>(storage_size)}, 
                                                     torch::TensorOptions().dtype(torch::kComplexFloat));
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}