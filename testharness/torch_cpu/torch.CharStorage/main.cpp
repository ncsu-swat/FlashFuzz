#include "fuzzer_utils.h" // General fuzzing utilities
#include <algorithm>
#include <cstring>
#include <iostream>

// Target API: torch.CharStorage

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

        // Seed tensor used to populate the CharStorage
        torch::Tensor seed = fuzzer_utils::createTensor(Data, Size, offset).to(torch::kChar);

        int64_t storage_size = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&storage_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        storage_size = std::abs(storage_size % 512) + 1;

        torch::Tensor char_tensor = torch::empty({storage_size}, torch::kChar);
        torch::Storage storage = char_tensor.storage();

        // Populate storage from the seed tensor or fuzz data
        torch::Tensor fill_tensor = torch::from_blob(storage.mutable_data(), {storage_size}, torch::kChar);
        int64_t copy_elems = std::min<int64_t>(seed.numel(), storage_size);
        if (copy_elems > 0) {
            std::memcpy(fill_tensor.data_ptr<int8_t>(), seed.data_ptr<int8_t>(), copy_elems);
        } else if (offset < Size) {
            fill_tensor.fill_(static_cast<int8_t>(Data[offset++]));
        }

        // Copy the storage into another CharStorage-sized buffer
        torch::Tensor copy_tensor = torch::zeros({storage_size}, torch::kChar);
        torch::Storage copy_storage = copy_tensor.storage();
        size_t bytes_to_copy = std::min(storage.nbytes(), copy_storage.nbytes());
        if (bytes_to_copy > 0) {
            std::memcpy(copy_storage.mutable_data(), storage.data(), bytes_to_copy);
        }

        // Create a CharStorage view from a blob of the fuzz data
        if (offset < Size) {
            int64_t blob_size = std::min<int64_t>(Size - offset, storage_size);
            torch::Tensor blob_tensor = torch::from_blob(
                const_cast<uint8_t*>(Data + offset),
                {blob_size},
                torch::kChar);
            torch::Storage blob_storage = blob_tensor.storage();
            volatile int8_t first = blob_storage.nbytes() > 0
                                        ? blob_tensor.data_ptr<int8_t>()[0]
                                        : 0;
            (void)first;
            offset += blob_size;
        }

        // Touch the data so the optimizer cannot elide the operations
        if (fill_tensor.numel() > 0) {
            volatile int8_t acc = fill_tensor[0].item<int8_t>();
            acc += static_cast<int8_t>(storage.nbytes() & 0x7F);
            (void)acc;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}