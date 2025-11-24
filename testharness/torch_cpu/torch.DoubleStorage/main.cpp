#include "fuzzer_utils.h" // General fuzzing utilities
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <iostream>

// Target API: torch.DoubleStorage

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

        torch::Tensor seed = fuzzer_utils::createTensor(Data, Size, offset).to(torch::kDouble);

        int64_t storage_size = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&storage_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        storage_size = std::clamp<int64_t>(std::abs(storage_size), 1, 512);

        torch::Tensor double_tensor = torch::empty({storage_size}, torch::kDouble);
        torch::Storage storage = double_tensor.storage();

        if (offset < Size) {
            uint8_t selector = Data[offset++] % 3;
            if (selector == 1 && seed.numel() > 0) {
                storage = seed.contiguous().storage();
            } else if (selector == 2) {
                int64_t blob_elems = std::min<int64_t>(
                    static_cast<int64_t>((Size - offset) / sizeof(double)),
                    storage_size);
                if (blob_elems > 0) {
                    torch::Tensor blob_tensor = torch::from_blob(
                        const_cast<uint8_t*>(Data + offset),
                        {blob_elems},
                        torch::kDouble);
                    storage = blob_tensor.storage();
                    offset += blob_elems * static_cast<int64_t>(sizeof(double));
                }
            }
        }

        int64_t available_elems = storage.nbytes() / static_cast<int64_t>(sizeof(double));
        available_elems = std::min<int64_t>(available_elems, 1024);

        if (available_elems > 0) {
            torch::Tensor fill_tensor = torch::from_blob(
                storage.mutable_data(),
                {available_elems},
                torch::kDouble);

            int64_t copy_elems = std::min<int64_t>(seed.numel(), available_elems);
            if (copy_elems > 0) {
                std::memcpy(fill_tensor.data_ptr<double>(), seed.data_ptr<double>(), copy_elems * sizeof(double));
            } else if (offset < Size) {
                double fill_value = static_cast<double>(Data[offset++]) / 255.0;
                fill_tensor.fill_(fill_value);
            }

            torch::Tensor copy_tensor = torch::zeros({available_elems}, torch::kDouble);
            torch::Storage copy_storage = copy_tensor.storage();
            size_t bytes_to_copy = std::min(storage.nbytes(), copy_storage.nbytes());
            if (bytes_to_copy > 0) {
                std::memcpy(copy_storage.mutable_data(), storage.data(), bytes_to_copy);
            }

            torch::Tensor view_tensor = torch::from_blob(
                storage.mutable_data(),
                {available_elems},
                torch::kDouble);
            volatile double first = view_tensor[0].item<double>();
            volatile double sum = copy_tensor.sum().item<double>();
            (void)first;
            (void)sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
