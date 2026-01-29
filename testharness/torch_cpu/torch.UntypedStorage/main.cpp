#include "fuzzer_utils.h"            // General fuzzing utilities
#include <c10/core/Allocator.h>      // For c10::GetAllocator
#include <algorithm>
#include <cstring>
#include <iostream>

// Target API: torch.UntypedStorage (keyword kept for harness checks)

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;

        if (Size < 4) {
            return 0;
        }

        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);

        // UntypedStorage maps to c10::Storage in C++.
        c10::Storage storage = tensor.storage();

        size_t storage_bytes = storage.nbytes();
        torch::Device device = storage.device();
        const void *data_ptr = storage_bytes > 0 ? storage.data() : nullptr;
        void *mutable_ptr = storage_bytes > 0 ? storage.mutable_data() : nullptr;
        volatile size_t touched = storage_bytes + (data_ptr != nullptr) + (mutable_ptr != nullptr);
        (void)touched;

        const size_t element_size = tensor.element_size();
        if (element_size > 0 && storage_bytes >= element_size) {
            int64_t view_elems = static_cast<int64_t>(
                std::min<size_t>(storage_bytes / element_size, 64));
            try {
                torch::Tensor storage_view =
                    torch::from_blob(storage.mutable_data(), {view_elems}, tensor.options());
                volatile double acc = storage_view.flatten().sum().item<double>();
                (void)acc;
            } catch (...) {
                // from_blob may fail for certain dtype/shape combinations
            }
        }

        torch::Tensor tensor_copy = tensor.clone();
        c10::Storage copied_storage = tensor_copy.storage();
        volatile size_t clone_bytes = copied_storage.nbytes();
        (void)clone_bytes;

        // Replace the storage buffer with a fresh allocation to exercise set_data_ptr.
        if (storage_bytes > 0) {
            try {
                c10::Allocator *allocator = c10::GetAllocator(device.type());
                size_t swap_bytes = std::min<size_t>(storage_bytes, 512);
                at::DataPtr new_ptr = allocator->allocate(swap_bytes);
                if (new_ptr) {
                    std::memset(new_ptr.get(), 0xAB, swap_bytes);
                    storage.set_data_ptr(std::move(new_ptr));
                    storage.set_nbytes(swap_bytes);
                }
            } catch (...) {
                // Allocator operations may fail for certain device types
            }
        }

        // Build a small standalone storage on CPU and view it as a tensor.
        if (offset < Size) {
            size_t requested = Data[offset++] % 64;
            size_t new_bytes = std::max<size_t>(requested, element_size > 0 ? element_size : 1);
            new_bytes = std::min<size_t>(new_bytes, 512);

            c10::Allocator *allocator = c10::GetAllocator(c10::DeviceType::CPU);
            c10::Storage fresh_storage(
                c10::Storage::use_byte_size_t{}, new_bytes, allocator, /*resizable=*/true);

            size_t copy_bytes = std::min<size_t>(new_bytes, Size - offset);
            if (copy_bytes > 0) {
                std::memcpy(fresh_storage.mutable_data(), Data + offset, copy_bytes);
                offset += copy_bytes;
            }

            size_t fresh_elems = element_size > 0 ? std::max<size_t>(1, new_bytes / element_size) : 1;
            fresh_elems = std::min<size_t>(fresh_elems, 32);
            try {
                torch::Tensor fresh_view = torch::from_blob(
                    fresh_storage.mutable_data(),
                    {static_cast<int64_t>(fresh_elems)},
                    tensor.options().device(torch::kCPU));
                volatile double check = fresh_view.flatten().sum().item<double>();
                (void)check;
            } catch (...) {
                // from_blob may fail for certain configurations
            }
        }

        #ifdef USE_CUDA
        if (torch::cuda::is_available()) {
            try {
                torch::Tensor cuda_tensor = tensor.to(torch::kCUDA, /*non_blocking=*/false, /*copy=*/true);
                c10::Storage cuda_storage = cuda_tensor.storage();
                volatile auto cuda_device = cuda_storage.device();
                (void)cuda_device;
            } catch (...) {
                // CUDA operations may fail
            }
        }
        #endif
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}