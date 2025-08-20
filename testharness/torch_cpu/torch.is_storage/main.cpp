#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <typeinfo>       // For type checking

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
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test storage operations on the tensor's storage
        auto storage = tensor.storage();
        
        // Check if storage is valid
        bool storage_defined = storage.defined();
        
        // Test with the tensor itself
        bool tensor_is_tensor = tensor.defined();
        
        // Create an empty storage and test it
        auto empty_storage = torch::Storage();
        bool empty_storage_defined = empty_storage.defined();
        
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
            bool sized_storage_defined = sized_storage.defined();
        }
        
        // Create a tensor from data and test its storage
        if (offset + 4 <= Size) {
            std::vector<float> data_vec(4);
            std::memcpy(data_vec.data(), Data + offset, 4 * sizeof(float));
            offset += 4 * sizeof(float);
            
            auto data_tensor = torch::from_blob(data_vec.data(), {4}, torch::kFloat);
            auto data_storage = data_tensor.storage();
            bool data_storage_defined = data_storage.defined();
        }
        
        // Test storage properties
        if (tensor.numel() > 0 && storage.defined()) {
            auto storage_nbytes = storage.nbytes();
            auto storage_device = storage.device();
            auto storage_dtype = storage.dtype();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}