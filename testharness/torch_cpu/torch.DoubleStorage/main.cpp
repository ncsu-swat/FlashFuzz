#include "fuzzer_utils.h" // General fuzzing utilities
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <iostream>

// Target API: torch.DoubleStorage

// --- Fuzzer Entry Point ---
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

        torch::Tensor seed = fuzzer_utils::createTensor(Data, Size, offset).to(torch::kDouble);

        int64_t storage_size = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&storage_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        storage_size = std::clamp<int64_t>(std::abs(storage_size), 1, 512);

        // Create a DoubleStorage via a double tensor
        torch::Tensor double_tensor = torch::empty({storage_size}, torch::kDouble);
        torch::Storage storage = double_tensor.storage();

        // Exercise storage properties
        volatile int64_t nbytes = storage.nbytes();
        volatile bool resizable = storage.resizable();
        volatile const void* data_ptr = storage.data();
        (void)nbytes;
        (void)resizable;
        (void)data_ptr;

        if (offset < Size) {
            uint8_t selector = Data[offset++] % 4;
            
            if (selector == 0) {
                // Use storage from seed tensor
                if (seed.numel() > 0) {
                    storage = seed.contiguous().storage();
                }
            } else if (selector == 1) {
                // Create storage with specific size and fill
                int64_t new_size = std::clamp<int64_t>(
                    static_cast<int64_t>(Data[offset % Size]) + 1, 1, 256);
                torch::Tensor new_tensor = torch::empty({new_size}, torch::kDouble);
                storage = new_tensor.storage();
                
                // Fill with pattern from fuzzer data
                double* ptr = static_cast<double*>(storage.mutable_data());
                for (int64_t i = 0; i < new_size && offset + i < Size; i++) {
                    ptr[i] = static_cast<double>(Data[(offset + i) % Size]) / 255.0;
                }
            } else if (selector == 2) {
                // Test storage from zeros/ones
                torch::Tensor zeros_tensor = torch::zeros({storage_size}, torch::kDouble);
                torch::Tensor ones_tensor = torch::ones({storage_size}, torch::kDouble);
                
                uint8_t choice = (offset < Size) ? Data[offset++] % 2 : 0;
                storage = (choice == 0) ? zeros_tensor.storage() : ones_tensor.storage();
            } else {
                // Test storage from randn (deterministic based on fuzzer seed)
                if (offset + sizeof(int64_t) <= Size) {
                    int64_t manual_seed;
                    std::memcpy(&manual_seed, Data + offset, sizeof(int64_t));
                    torch::manual_seed(manual_seed);
                }
                torch::Tensor randn_tensor = torch::randn({storage_size}, torch::kDouble);
                storage = randn_tensor.storage();
            }
        }

        int64_t available_elems = storage.nbytes() / static_cast<int64_t>(sizeof(double));
        available_elems = std::min<int64_t>(available_elems, 1024);

        if (available_elems > 0) {
            // Create a view tensor from storage
            torch::Tensor view_tensor = torch::from_blob(
                storage.mutable_data(),
                {available_elems},
                torch::kDouble);

            // Test various operations on storage-backed tensor
            try {
                int64_t copy_elems = std::min<int64_t>(seed.numel(), available_elems);
                if (copy_elems > 0) {
                    // Copy data from seed to storage
                    std::memcpy(view_tensor.data_ptr<double>(), 
                               seed.data_ptr<double>(), 
                               copy_elems * sizeof(double));
                } else if (offset < Size) {
                    // Fill with constant value
                    double fill_value = static_cast<double>(Data[offset++]) / 255.0;
                    view_tensor.fill_(fill_value);
                }
            } catch (...) {
                // Silently catch shape/size mismatches
            }

            // Create a copy via separate storage
            torch::Tensor copy_tensor = torch::zeros({available_elems}, torch::kDouble);
            torch::Storage copy_storage = copy_tensor.storage();
            
            size_t bytes_to_copy = std::min(storage.nbytes(), copy_storage.nbytes());
            if (bytes_to_copy > 0) {
                std::memcpy(copy_storage.mutable_data(), storage.data(), bytes_to_copy);
            }

            // Exercise storage comparison
            volatile bool is_alias = storage.is_alias_of(copy_storage);
            (void)is_alias;

            // Read values to ensure storage is valid
            volatile double first = view_tensor[0].item<double>();
            volatile double sum = copy_tensor.sum().item<double>();
            (void)first;
            (void)sum;

            // Test storage device info
            volatile auto device = storage.device();
            (void)device;
        }

        // Test storage set_data_ptr functionality via tensor operations
        if (offset + 1 < Size && storage_size > 1) {
            try {
                uint8_t op = Data[offset++] % 3;
                torch::Tensor test_tensor = torch::from_blob(
                    storage.mutable_data(),
                    {std::min<int64_t>(storage_size, available_elems)},
                    torch::kDouble);
                
                if (op == 0) {
                    // In-place add
                    test_tensor.add_(1.0);
                } else if (op == 1) {
                    // In-place mul
                    test_tensor.mul_(2.0);
                } else {
                    // In-place fill
                    test_tensor.zero_();
                }
                
                // Verify the storage was modified
                volatile double check = test_tensor[0].item<double>();
                (void)check;
            } catch (...) {
                // Silently catch any operation failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}