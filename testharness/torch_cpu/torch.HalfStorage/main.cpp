#include "fuzzer_utils.h" // General fuzzing utilities
#include <algorithm>
#include <cstring>
#include <iostream> // For cerr

// Target API: torch.HalfStorage

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;

        if (Size < 2) {
            return 0;
        }

        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        if (tensor.dtype() != torch::kHalf) {
            tensor = tensor.to(torch::kHalf);
        }
        tensor = tensor.contiguous();

        torch::Storage storage = tensor.storage();
        size_t nbytes = storage.nbytes();
        size_t num_elems = nbytes / sizeof(at::Half);
        const void *raw_data = storage.data();

        if (num_elems > 0 && raw_data) {
            const at::Half *half_ptr = static_cast<const at::Half *>(raw_data);
            volatile at::Half first = half_ptr[0];
            size_t idx = 0;
            if (Size > offset) {
                idx = Data[offset++] % num_elems;
            }
            volatile at::Half sample = half_ptr[idx];
            (void)first;
            (void)sample;
        }

        size_t capped = std::min<size_t>(num_elems, 256);
        if (capped > 0 && raw_data) {
            auto copy_tensor = torch::zeros({static_cast<int64_t>(capped)}, torch::kHalf);
            auto copy_storage = copy_tensor.storage();
            size_t copy_bytes = std::min(copy_storage.nbytes(), nbytes);
            std::memcpy(copy_storage.mutable_data(), raw_data, copy_bytes);
            volatile at::Half echo = copy_tensor[0].item<at::Half>();
            (void)echo;
        }

        if (num_elems > 0 && offset + sizeof(uint16_t) <= Size) {
            uint16_t raw_val;
            std::memcpy(&raw_val, Data + offset, sizeof(uint16_t));
            offset += sizeof(uint16_t);
            at::Half fill;
            std::memcpy(&fill, &raw_val, sizeof(uint16_t));

            auto writable = static_cast<at::Half *>(storage.mutable_data());
            size_t fill_count = std::min<size_t>(num_elems, 512);
            for (size_t i = 0; i < fill_count; ++i) {
                writable[i] = fill;
            }
        }

        if (num_elems > 0) {
            auto view_tensor = torch::from_blob(
                storage.mutable_data(),
                {static_cast<int64_t>(num_elems)},
                torch::TensorOptions().dtype(torch::kHalf));
            volatile at::Half tail = view_tensor[view_tensor.numel() - 1].item<at::Half>();
            (void)tail;
        }

        if (offset < Size) {
            size_t remaining_bytes = Size - offset;
            size_t extra_elems = std::min<size_t>(remaining_bytes / sizeof(at::Half), 128);
            if (extra_elems > 0) {
                auto data_tensor = torch::empty({static_cast<int64_t>(extra_elems)}, torch::kHalf);
                std::memcpy(data_tensor.data_ptr(), Data + offset, extra_elems * sizeof(at::Half));
                torch::Storage data_storage = data_tensor.storage();
                volatile size_t touched = data_storage.nbytes();
                (void)touched;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
