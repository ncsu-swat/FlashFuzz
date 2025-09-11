#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <sstream>        // For stringstream

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to serialize
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a serialization context
        torch::serialize::OutputArchive output_archive;
        
        // Serialize the tensor
        output_archive.write("tensor", tensor);
        
        // Get the serialized data using stringstream
        std::stringstream ss;
        output_archive.save_to(ss);
        std::string serialized_data = ss.str();
        
        // Create a deserialization storage context
        c10::intrusive_ptr<torch::jit::PyTorchStreamReader> stream_reader = 
            c10::make_intrusive<torch::jit::PyTorchStreamReader>(
                std::make_shared<torch::jit::MemoryReadAdapter>(
                    serialized_data.data(), serialized_data.size()));
        
        // Create a deserialization storage context
        auto storage_context = c10::make_intrusive<torch::jit::DeserializationStorageContext>();
        
        // Create an input archive with the storage context
        torch::serialize::InputArchive input_archive(std::move(stream_reader), storage_context);
        
        // Deserialize the tensor
        torch::Tensor deserialized_tensor;
        input_archive.read("tensor", deserialized_tensor);
        
        // Try to access the deserialized tensor to ensure it works
        auto sizes = deserialized_tensor.sizes();
        auto dtype = deserialized_tensor.dtype();
        
        // Try to perform an operation on the deserialized tensor
        if (deserialized_tensor.numel() > 0) {
            auto sum = deserialized_tensor.sum();
        }
        
        // Try to modify the storage context
        if (offset + 1 < Size) {
            uint8_t key_length = Data[offset++];
            if (key_length > 0 && offset + key_length <= Size) {
                std::string key(reinterpret_cast<const char*>(Data + offset), key_length);
                offset += key_length;
                
                // Try to get a non-existent storage from the context
                try {
                    auto storage = storage_context->getStorage(key);
                } catch (...) {
                    // Expected to fail for non-existent keys
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
