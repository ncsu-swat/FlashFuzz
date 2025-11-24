#include "fuzzer_utils.h" // General fuzzing utilities
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>

// Target API: torch.IntStorage
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;

        if (Size < 4) {
            return 0;
        }

        torch::Tensor seed = fuzzer_utils::createTensor(Data, Size, offset).to(torch::kInt);

        int64_t storage_elems = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&storage_elems, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        storage_elems = std::abs(storage_elems % 256) + 1;

        torch::Tensor int_tensor = torch::zeros({storage_elems}, torch::kInt);
        torch::Storage storage = int_tensor.storage();

        int64_t copy_elems = std::min<int64_t>(seed.numel(), storage_elems);
        if (copy_elems > 0) {
            auto seed_contiguous = seed.contiguous();
            std::memcpy(
                storage.mutable_data(),
                seed_contiguous.data_ptr<int>(),
                static_cast<size_t>(copy_elems) * sizeof(int));
        } else if (storage_elems > 0 && offset < Size) {
            int32_t fill_value = static_cast<int8_t>(Data[offset++]);
            auto data_ptr = static_cast<int32_t *>(storage.mutable_data());
            for (int64_t i = 0; i < storage_elems; ++i) {
                data_ptr[i] = fill_value;
            }
        }

        size_t nbytes = storage.nbytes();
        if (nbytes > 0) {
            const int32_t *data_ptr = static_cast<const int32_t *>(storage.data());
            volatile int32_t first = data_ptr[0];
            volatile int32_t last = data_ptr[(nbytes / sizeof(int32_t)) - 1];
            (void)first;
            (void)last;
        }

        int64_t view_elems = std::max<int64_t>(1, std::min<int64_t>(storage_elems, 32));
        torch::Tensor view_tensor = torch::from_blob(
            storage.mutable_data(),
            {view_elems},
            torch::TensorOptions().dtype(torch::kInt));
        torch::Storage view_storage = view_tensor.storage();
        if (view_storage.nbytes() >= sizeof(int32_t)) {
            auto view_ptr = static_cast<int32_t *>(view_storage.mutable_data());
            view_ptr[0] = static_cast<int32_t>(view_storage.nbytes());
        }

        torch::Tensor copy_tensor = torch::zeros({storage_elems}, torch::kInt);
        torch::Storage copy_storage = copy_tensor.storage();
        size_t bytes_to_copy = std::min<size_t>(copy_storage.nbytes(), nbytes);
        if (bytes_to_copy > 0) {
            std::memcpy(copy_storage.mutable_data(), storage.data(), bytes_to_copy);
        }

        if (offset < Size) {
            int64_t blob_elems = std::min<int64_t>(
                static_cast<int64_t>((Size - offset) / sizeof(int32_t)),
                static_cast<int64_t>(64));
            if (blob_elems > 0) {
                torch::Tensor blob_tensor = torch::from_blob(
                    const_cast<uint8_t *>(Data + offset),
                    {blob_elems},
                    torch::TensorOptions().dtype(torch::kInt));
                torch::Storage blob_storage = blob_tensor.storage();
                volatile int32_t blob_sample = blob_storage.nbytes() > 0
                                                   ? blob_tensor.data_ptr<int32_t>()[0]
                                                   : 0;
                (void)blob_sample;
                offset += static_cast<size_t>(blob_elems * sizeof(int32_t));
            }
        }

        if (storage_elems > 0) {
            torch::Tensor tensor_from_storage = torch::from_blob(
                storage.mutable_data(),
                {storage_elems},
                torch::TensorOptions().dtype(torch::kInt));
            volatile int32_t touch = tensor_from_storage[0].item<int32_t>();
            (void)touch;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
