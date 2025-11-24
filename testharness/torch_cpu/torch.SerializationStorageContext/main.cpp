#include "fuzzer_utils.h"                                   // General fuzzing utilities
#include <torch/csrc/jit/serialization/storage_context.h>   // torch::jit::SerializationStorageContext
#include <iostream>                                         // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        if (Size < 1)
        {
            return 0;
        }

        torch::jit::SerializationStorageContext context;

        // Create a tensor to be serialized
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        const auto &storage = tensor.storage();

        const auto storageKey = context.getOrAddStorage(storage);
        const auto storageKeyRepeat = context.getOrAddStorage(storage);
        if (storageKey != storageKeyRepeat || !context.hasStorage(storage))
        {
            throw std::runtime_error("Storage lookup failed");
        }

        // Create another tensor with different data (bounded by fuzzer_utils helpers)
        torch::Tensor anotherTensor =
            (offset < Size) ? fuzzer_utils::createTensor(Data, Size, offset) : torch::ones({2, 3});
        const auto &anotherStorage = anotherTensor.storage();
        const auto anotherKey = context.getOrAddStorage(anotherStorage);
        if (!context.hasStorage(anotherStorage))
        {
            throw std::runtime_error("Secondary storage missing");
        }

        // Clone to force a distinct storage and ensure it also registers
        torch::Tensor clonedTensor = tensor.clone();
        const auto &clonedStorage = clonedTensor.storage();
        const auto clonedKey = context.getOrAddStorage(clonedStorage);
        if (!context.hasStorage(clonedStorage))
        {
            throw std::runtime_error("Cloned storage not tracked");
        }

        // Touch storages so the fuzzer drives through allocation paths
        volatile uint8_t touch = 0;
        touch ^= static_cast<uint8_t>(storageKey);
        touch ^= static_cast<uint8_t>(anotherKey);
        touch ^= static_cast<uint8_t>(clonedKey);
        (void)touch;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
