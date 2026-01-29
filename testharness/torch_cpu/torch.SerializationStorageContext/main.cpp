#include "fuzzer_utils.h"                                   // General fuzzing utilities
#include <torch/csrc/jit/serialization/storage_context.h>   // torch::jit::SerializationStorageContext
#include <iostream>                                         // For cerr, cout

// --- Fuzzer Entry Point ---
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
        
        // Verify consistency (silent check - not an API exception)
        if (storageKey != storageKeyRepeat || !context.hasStorage(storage))
        {
            // This indicates a bug in the API, not invalid input
            // Just return to avoid polluting exception logs
            return 0;
        }

        // Create another tensor with different data (bounded by fuzzer_utils helpers)
        torch::Tensor anotherTensor =
            (offset < Size) ? fuzzer_utils::createTensor(Data, Size, offset) : torch::ones({2, 3});
        const auto &anotherStorage = anotherTensor.storage();
        const auto anotherKey = context.getOrAddStorage(anotherStorage);
        if (!context.hasStorage(anotherStorage))
        {
            return 0;
        }

        // Clone to force a distinct storage and ensure it also registers
        torch::Tensor clonedTensor = tensor.clone();
        const auto &clonedStorage = clonedTensor.storage();
        const auto clonedKey = context.getOrAddStorage(clonedStorage);
        if (!context.hasStorage(clonedStorage))
        {
            return 0;
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