#include "fuzzer_utils.h" // General fuzzing utilities
#include <algorithm>
#include <cstring>
#include <iostream> // For cerr
#include <tuple>    // For std::get with lu_unpack result

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
        
        if (Size < 2) {
            return 0;
        }
        
        // Target API: torch::Storage (C++ equivalent of torch.FloatStorage)
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        if (tensor.dtype() != torch::kFloat) {
            tensor = tensor.to(torch::kFloat);
        }

        auto storage = tensor.storage();
        size_t storage_elems = storage.nbytes() / sizeof(float);
        const void *raw_data = storage.data();
        auto nbytes = storage.nbytes();
        auto device = storage.device();
        (void)device; // silence unused variable in optimized builds

        if (storage_elems > 0) {
            const float *data_ptr = static_cast<const float *>(raw_data);
            float first_element = data_ptr[0];
            float last_element = data_ptr[storage_elems - 1];
            (void)first_element;
            (void)last_element;

            if (storage_elems > 1 && offset < Size) {
                size_t idx = Data[offset++] % storage_elems;
                auto random_element = data_ptr[idx];
                (void)random_element;
            }
        }

        size_t capped_elems = std::min<size_t>(storage_elems, 128);
        if (capped_elems > 0 && nbytes > 0) {
            auto new_tensor = torch::zeros({static_cast<int64_t>(capped_elems)}, torch::kFloat);
            auto new_storage = new_tensor.storage();
            size_t bytes_to_copy = std::min<size_t>(nbytes, capped_elems * sizeof(float));
            std::memcpy(new_storage.mutable_data(), raw_data, bytes_to_copy);

            size_t partial_elems = std::max<size_t>(static_cast<size_t>(1), capped_elems / 2);
            auto partial_tensor = torch::zeros({static_cast<int64_t>(partial_elems)}, torch::kFloat);
            auto partial_storage = partial_tensor.storage();
            size_t partial_bytes = std::min<size_t>(nbytes, partial_elems * sizeof(float));
            std::memcpy(partial_storage.mutable_data(), raw_data, partial_bytes);
        }

        if (offset + sizeof(float) <= Size && storage_elems > 0) {
            float fill_value;
            std::memcpy(&fill_value, Data + offset, sizeof(float));
            offset += sizeof(float);

            auto writable_ptr = static_cast<float *>(storage.mutable_data());
            size_t fill_count = std::min<size_t>(storage_elems, 256);
            for (size_t i = 0; i < fill_count; ++i) {
                writable_ptr[i] = fill_value;
            }
        }

        if (storage_elems > 0) {
            auto tensor_from_storage = torch::from_blob(
                storage.mutable_data(),
                {static_cast<int64_t>(storage_elems)},
                torch::TensorOptions().dtype(torch::kFloat));
            (void)tensor_from_storage;
        }

        if (offset < Size) {
            size_t custom_size = std::max<size_t>(static_cast<size_t>(1), static_cast<size_t>(Data[offset++] % 64));
            auto custom_tensor = torch::zeros({static_cast<int64_t>(custom_size)}, torch::kFloat);
            auto custom_storage = custom_tensor.storage();
            (void)custom_storage;
        }

        if (storage_elems > 0) {
            auto storage_from_data = torch::from_blob(
                storage.mutable_data(),
                {static_cast<int64_t>(storage_elems)},
                torch::TensorOptions().dtype(torch::kFloat));
            (void)storage_from_data;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}