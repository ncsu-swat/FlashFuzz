#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        // Need at least a few bytes to create a meaningful test
        if (Size < 4) {
            return 0;
        }

        size_t offset = 0;

        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);

        // Get the number of elements in the tensor
        int64_t numel = tensor.numel();
        if (numel <= 0) {
            return 0;
        }

        // 1. Create a ByteStorage by creating a byte tensor
        torch::Tensor byte_tensor = torch::empty({numel}, torch::kByte);
        torch::Storage storage = byte_tensor.storage();

        // 2. Convert input tensor to byte type and copy to storage
        try {
            torch::Tensor converted_tensor = tensor.to(torch::kByte);
            // Copy data if sizes match
            int64_t copy_size = std::min(converted_tensor.numel(), byte_tensor.numel());
            if (copy_size > 0) {
                std::memcpy(byte_tensor.data_ptr<uint8_t>(),
                           converted_tensor.data_ptr<uint8_t>(),
                           copy_size);
            }
        } catch (...) {
            // Conversion may fail for some tensor types, continue with empty tensor
        }

        // 3. Test storage size
        int64_t storage_nbytes = storage.nbytes();

        // 4. Test storage data access
        if (storage_nbytes > 0 && byte_tensor.numel() > 0) {
            // Access via tensor to ensure proper memory management
            uint8_t first_byte = byte_tensor.data_ptr<uint8_t>()[0];
            (void)first_byte;

            // Test setting values via tensor
            byte_tensor.data_ptr<uint8_t>()[0] = 255;

            if (byte_tensor.numel() > 1) {
                byte_tensor.data_ptr<uint8_t>()[byte_tensor.numel() - 1] = 128;
            }
        }

        // 5. Test creating storage with different sizes
        if (offset + sizeof(int32_t) <= Size) {
            int32_t new_size_raw;
            std::memcpy(&new_size_raw, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);

            // Make sure new_size is reasonable (1 to 500)
            int64_t new_size = (std::abs(new_size_raw) % 500) + 1;

            torch::Tensor resized_tensor = torch::empty({new_size}, torch::kByte);
            torch::Storage resized_storage = resized_tensor.storage();

            // 6. Test filling tensor
            if (offset < Size) {
                uint8_t fill_value = Data[offset++];
                resized_tensor.fill_(fill_value);
            }

            // 7. Test storage copy
            torch::Tensor copy_tensor = torch::empty({new_size}, torch::kByte);
            copy_tensor.copy_(resized_tensor);
            torch::Storage storage_copy = copy_tensor.storage();

            // 8. Test storage properties
            torch::Device device = resized_storage.device();
            (void)device;

            // 9. Test creating a view/slice of the tensor
            if (new_size > 2 && offset + sizeof(int16_t) <= Size) {
                int16_t offset_raw;
                std::memcpy(&offset_raw, Data + offset, sizeof(int16_t));
                offset += sizeof(int16_t);

                int64_t slice_start = std::abs(offset_raw) % (new_size - 1);
                int64_t slice_size = new_size - slice_start;

                if (slice_size > 0) {
                    torch::Tensor view_tensor = resized_tensor.narrow(0, slice_start, slice_size);
                    torch::Storage view_storage = view_tensor.storage();
                    (void)view_storage;
                }
            }

            // 10. Test contiguous storage access
            torch::Tensor contig_tensor = resized_tensor.contiguous();
            torch::Storage contig_storage = contig_tensor.storage();

            // 11. Test storage data pointer access (via tensor)
            const uint8_t* const_data = contig_tensor.data_ptr<uint8_t>();
            uint8_t* mutable_data = contig_tensor.data_ptr<uint8_t>();
            (void)const_data;
            (void)mutable_data;

            // 12. Test storage is_alias_of
            bool is_alias = contig_storage.is_alias_of(resized_storage);
            (void)is_alias;
        }

        // 13. Test operations that use byte storage
        if (offset + sizeof(int16_t) <= Size) {
            int16_t size_raw;
            std::memcpy(&size_raw, Data + offset, sizeof(int16_t));
            offset += sizeof(int16_t);

            int64_t test_size = (std::abs(size_raw) % 100) + 1;

            // Create byte tensors for various operations
            torch::Tensor t1 = torch::randint(0, 256, {test_size}, torch::kByte);
            torch::Tensor t2 = torch::randint(0, 256, {test_size}, torch::kByte);

            // Test clone (creates new storage)
            torch::Tensor cloned = t1.clone();
            torch::Storage cloned_storage = cloned.storage();

            // Test storage sharing detection
            torch::Tensor shared_view = t1.view({-1});
            bool shares_storage = t1.storage().is_alias_of(shared_view.storage());
            (void)shares_storage;

            // Test storage after reshape
            torch::Tensor reshaped = t1.reshape({test_size});
            torch::Storage reshaped_storage = reshaped.storage();
            (void)reshaped_storage;

            // 14. Test zeros and ones with byte type
            torch::Tensor zeros_byte = torch::zeros({test_size}, torch::kByte);
            torch::Tensor ones_byte = torch::ones({test_size}, torch::kByte);
            (void)zeros_byte;
            (void)ones_byte;

            // 15. Test storage after arithmetic (converted to appropriate type)
            try {
                torch::Tensor sum_result = t1.to(torch::kInt32) + t2.to(torch::kInt32);
                torch::Tensor byte_result = sum_result.clamp(0, 255).to(torch::kByte);
                torch::Storage result_storage = byte_result.storage();
                (void)result_storage;
            } catch (...) {
                // Some operations may fail, that's ok
            }
        }

        // 16. Test CPU storage explicitly
        torch::Tensor cpu_tensor = torch::empty({10}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCPU));
        torch::Storage cpu_storage = cpu_tensor.storage();
        (void)cpu_storage;

    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}