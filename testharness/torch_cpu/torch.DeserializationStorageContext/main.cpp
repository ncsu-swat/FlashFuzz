#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <sstream>        // For stringstream
#include <algorithm>      // For std::min
#include <memory>         // For std::make_shared
#include <torch/csrc/jit/serialization/storage_context.h>

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
        auto storage_context = std::make_shared<torch::jit::DeserializationStorageContext>();
        
        // Create an input archive and load from serialized buffer
        torch::serialize::InputArchive input_archive;
        input_archive.load_from(serialized_data.data(), serialized_data.size());
        
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
        
        // Add the tensor's storage to the context and optionally another fuzzed key
        const std::string base_key = "tensor_storage";
        if (deserialized_tensor.defined() && deserialized_tensor.has_storage() &&
            !storage_context->hasStorage(base_key)) {
            storage_context->addStorage(base_key, deserialized_tensor.storage());
        }

        if (offset < Size) {
            uint8_t key_length = Data[offset++];
            key_length = std::min<uint8_t>(key_length, 32);
            if (key_length > 0 && offset + key_length <= Size &&
                deserialized_tensor.defined() && deserialized_tensor.has_storage()) {
                std::string fuzz_key(reinterpret_cast<const char*>(Data + offset), key_length);
                offset += key_length;

                // Ensure unique key before adding
                if (!storage_context->hasStorage(fuzz_key)) {
                    storage_context->addStorage(fuzz_key, deserialized_tensor.clone().storage());
                }
            }
        }

        // Touch storages to exercise DeserializationStorageContext paths
        if (storage_context->hasStorage(base_key)) {
            auto storage = storage_context->getStorage(base_key);
            auto impl = storage.unsafeGetStorageImpl();
            (void)impl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
