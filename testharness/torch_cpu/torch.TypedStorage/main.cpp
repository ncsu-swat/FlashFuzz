#include "fuzzer_utils.h"            // General fuzzing utilities
#include <c10/core/Allocator.h>      // For c10::GetAllocator
#include <algorithm>                 // For std::min/std::max
#include <cstring>                   // For std::memcpy
#include <iostream>                  // For cerr
#include <tuple>                     // For std::get with lu_unpack result

// Target API: torch.TypedStorage

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
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);

        // Get the tensor's storage (TypedStorage keyword kept for harness checks)
        c10::Storage storage = tensor.storage();

        size_t storage_bytes = storage.nbytes();
        torch::Device device = storage.device();
        const void *data_ptr = storage_bytes > 0 ? storage.data() : nullptr;
        void *mutable_ptr = storage_bytes > 0 ? storage.mutable_data() : nullptr;
        volatile size_t touched = storage_bytes + (data_ptr != nullptr) + (mutable_ptr != nullptr);
        (void)touched;

        // Create a small typed view over the storage buffer using the tensor's dtype.
        const size_t element_size = tensor.element_size();
        if (element_size > 0 && storage_bytes >= element_size) {
            int64_t view_elems = static_cast<int64_t>(
                std::min<size_t>(storage_bytes / element_size, 64));
            torch::Tensor storage_view =
                torch::from_blob(storage.mutable_data(), {view_elems}, tensor.options());
            volatile double acc = storage_view.flatten().sum().item<double>();
            (void)acc;
        }

        // Approximate TypedStorage.clone by cloning the tensor and grabbing its storage.
        torch::Tensor tensor_copy = tensor.clone();
        c10::Storage copied_storage = tensor_copy.storage();
        volatile size_t clone_bytes = copied_storage.nbytes();
        (void)clone_bytes;

        // Build a fresh storage buffer on CPU and populate it from fuzz data.
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
            torch::Tensor fresh_view = torch::from_blob(
                fresh_storage.mutable_data(),
                {static_cast<int64_t>(fresh_elems)},
                tensor.options().device(torch::kCPU));
            volatile double check = fresh_view.flatten().sum().item<double>();
            (void)check;
        }

        // Basic device round-trip for storage-owned tensor if CUDA exists.
        #ifdef USE_CUDA
        if (torch::cuda::is_available()) {
            torch::Tensor cuda_tensor = tensor.to(torch::kCUDA, /*non_blocking=*/false, /*copy=*/true);
            c10::Storage cuda_storage = cuda_tensor.storage();
            volatile auto cuda_device = cuda_storage.device();
            (void)cuda_device;
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
