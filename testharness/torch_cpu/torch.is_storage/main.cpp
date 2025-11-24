#include "fuzzer_utils.h" // General fuzzing utilities
#include <cstring>        // For memcpy
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <typeinfo>       // For type checking

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
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
        // torch.is_storage target keyword anchor
        auto storage = tensor.storage();
        
        // Check if storage is valid
        bool storage_defined = static_cast<bool>(storage);
        
        // Test with the tensor itself
        bool tensor_is_tensor = tensor.defined();
        (void)tensor_is_tensor;
        
        // Create an empty storage and test it
        auto empty_storage = torch::Storage();
        bool empty_storage_defined = static_cast<bool>(empty_storage);
        (void)empty_storage_defined;
        
        // Create a storage with specific size and test it
        if (offset < Size) {
            uint8_t size_byte = Data[offset++];
            size_t storage_size = size_byte % 100; // Limit size to avoid excessive memory usage
            
            auto sized_storage = torch::Storage(
                torch::Storage::use_byte_size_t{},
                storage_size,
                nullptr,
                false
            );
            bool sized_storage_defined = static_cast<bool>(sized_storage);
            (void)sized_storage_defined;
        }
        
        // Create a tensor from data and test its storage
        if (offset + 4 * sizeof(float) <= Size) {
            std::vector<float> data_vec(4);
            std::memcpy(data_vec.data(), Data + offset, 4 * sizeof(float));
            offset += 4 * sizeof(float);
            
            auto data_tensor = torch::from_blob(data_vec.data(), {4}, torch::kFloat).clone();
            auto data_storage = data_tensor.storage();
            bool data_storage_defined = static_cast<bool>(data_storage);
            (void)data_storage_defined;
        }
        
        // Test storage properties
        if (tensor.numel() > 0 && storage_defined) {
            auto storage_nbytes = storage.nbytes();
            auto storage_device = storage.device();
            const void *storage_data = storage.data();
            bool is_storage_like = static_cast<bool>(storage);
            (void)storage_nbytes;
            (void)storage_device;
            (void)storage_data;
            (void)is_storage_like;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
