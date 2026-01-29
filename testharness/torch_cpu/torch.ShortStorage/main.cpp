#include "fuzzer_utils.h"        // General fuzzing utilities
#include <c10/core/CPUAllocator.h> // For c10::GetDefaultCPUAllocator
#include <algorithm>             // For std::min
#include <cmath>                 // For std::abs
#include <cstring>               // For std::memcpy
#include <cstdint>               // For fixed-width integer types
#include <iostream>              // For cerr
#include <tuple>                 // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    // Target API keyword: torch.ShortStorage
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
            auto short_tensor = tensor.to(torch::kInt16).reshape(-1).contiguous();
            constexpr int64_t kMaxElements = 4096;
            const int64_t storage_elements = std::min<int64_t>(short_tensor.numel(), kMaxElements);
            const size_t storage_bytes = static_cast<size_t>(storage_elements) * sizeof(int16_t);
            
            c10::Storage storage(
                c10::Storage::use_byte_size_t(),
                storage_bytes,
                c10::GetDefaultCPUAllocator(),
                true);
            
            int16_t *storage_data = static_cast<int16_t *>(storage.mutable_data());
            if (storage_data && storage_elements > 0) {
                auto tensor_accessor = short_tensor.accessor<int16_t, 1>();
                for (int64_t i = 0; i < storage_elements; i++) {
                    storage_data[i] = tensor_accessor[i];
                }
            }
            
            int64_t storage_size = static_cast<int64_t>(storage.nbytes() / sizeof(int16_t));
            int32_t accumulator = 0;
            if (storage_data && storage_size > 0) {
                accumulator += storage_data[0];
                
                // Test element assignment if size allows
                if (storage_size > 1) {
                    storage_data[1] = 42;
                    accumulator += storage_data[1];
                }
            }
            
            // Test copy constructor
            auto storage_copy = storage;
            
            // Test move constructor
            auto storage_moved = std::move(storage_copy);
            
            // Test storage properties
            (void)storage_moved.nbytes();
            (void)storage_moved.resizable();
            (void)storage_moved.data_ptr();
            
            // Test resize by reallocating through the allocator when possible
            if (offset + sizeof(int64_t) <= Size && storage_moved.resizable() && storage_moved.allocator()) {
                int64_t new_size_raw;
                std::memcpy(&new_size_raw, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                const int64_t new_elements = std::min<int64_t>(std::abs(new_size_raw) % (kMaxElements + 1), kMaxElements);
                const size_t new_bytes = static_cast<size_t>(new_elements) * sizeof(int16_t);
                if (new_bytes > 0) {
                    auto new_ptr = storage_moved.allocator()->allocate(new_bytes);
                    storage_moved.set_data_ptr_noswap(std::move(new_ptr));
                    storage_moved.set_nbytes(new_bytes);
                }
                storage_data = static_cast<int16_t *>(storage_moved.mutable_data());
                storage_size = static_cast<int64_t>(storage_moved.nbytes() / sizeof(int16_t));
            }
            
            // Test fill
            if (offset < Size && storage_data && storage_size > 0) {
                int16_t fill_value = static_cast<int16_t>(Data[offset++]);
                for (int64_t i = 0; i < storage_size; i++) {
                    storage_data[i] = fill_value;
                }
            }
            
            // Create a tensor from the storage and exercise it
            if (storage_size > 0) {
                auto tensor_from_storage = torch::empty({storage_size}, torch::kInt16);
                if (storage_data) {
                    auto accessor = tensor_from_storage.accessor<int16_t, 1>();
                    for (int64_t i = 0; i < storage_size; i++) {
                        accessor[i] = storage_data[i];
                    }
                }
                auto reduction = tensor_from_storage.sum().item<int64_t>() + accumulator;
                (void)reduction;
            }
            
            // Test zero-sized storage creation
            if (offset < Size && (Data[offset] % 4 == 0)) {
                c10::Storage empty_storage(
                    c10::Storage::use_byte_size_t(),
                    0,
                    c10::GetDefaultCPUAllocator(),
                    true);
                (void)empty_storage.nbytes();
            }
        }
        catch (const c10::Error &e) {
            // PyTorch specific errors are expected and okay
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}