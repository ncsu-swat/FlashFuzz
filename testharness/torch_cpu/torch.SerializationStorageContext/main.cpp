#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to be serialized
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a serialization storage context using torch::jit
        torch::jit::SerializationStorageContext context;
        
        // Get a unique ID for the tensor's storage
        const auto& storage = tensor.storage();
        const auto storageKey = context.getOrAddStorage(storage);
        
        // Test if we can retrieve the storage back from the context
        if (context.hasStorage(storageKey)) {
            auto retrievedStorage = context.getStorage(storageKey);
            
            // Verify the retrieved storage matches the original
            if (retrievedStorage.data() != storage.data() || 
                retrievedStorage.nbytes() != storage.nbytes() ||
                retrievedStorage.device().type() != storage.device().type()) {
                throw std::runtime_error("Retrieved storage doesn't match original");
            }
        }
        
        // Test adding a storage with a custom key
        uint64_t customKey = 0;
        if (offset + sizeof(uint64_t) <= Size) {
            std::memcpy(&customKey, Data + offset, sizeof(uint64_t));
            offset += sizeof(uint64_t);
        }
        
        // Create another tensor with different data
        torch::Tensor anotherTensor;
        if (offset < Size) {
            anotherTensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            anotherTensor = torch::ones({2, 3});
        }
        
        // Add the storage with the custom key
        context.addStorage(customKey, anotherTensor.storage());
        
        // Verify we can retrieve it
        if (context.hasStorage(customKey)) {
            auto retrievedStorage = context.getStorage(customKey);
            
            // Verify the retrieved storage matches
            if (retrievedStorage.data() != anotherTensor.storage().data() || 
                retrievedStorage.nbytes() != anotherTensor.storage().nbytes() ||
                retrievedStorage.device().type() != anotherTensor.storage().device().type()) {
                throw std::runtime_error("Retrieved custom storage doesn't match original");
            }
        }
        
        // Test with empty tensor
        torch::Tensor emptyTensor = torch::empty({0});
        const auto emptyStorageKey = context.getOrAddStorage(emptyTensor.storage());
        
        // Test with scalar tensor
        torch::Tensor scalarTensor = torch::tensor(3.14);
        const auto scalarStorageKey = context.getOrAddStorage(scalarTensor.storage());
        
        // Test with boolean tensor
        torch::Tensor boolTensor = torch::tensor(true);
        const auto boolStorageKey = context.getOrAddStorage(boolTensor.storage());
        
        // Test with complex tensor if available
        torch::Tensor complexTensor = torch::complex(torch::rand({2, 2}), torch::rand({2, 2}));
        const auto complexStorageKey = context.getOrAddStorage(complexTensor.storage());
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}