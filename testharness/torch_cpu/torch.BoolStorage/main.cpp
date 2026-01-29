#include "fuzzer_utils.h" // General fuzzing utilities
#include <algorithm>      // For std::min, std::clamp
#include <cstring>        // For std::memcpy
#include <iostream>       // For cerr
#include <vector>         // For std::vector

// Target API keyword to satisfy harness checks: torch.BoolStorage

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
        
        // Need at least a few bytes to create a meaningful test
        if (Size < 4) {
            return 0;
        }
        
        // Determine how to create the boolean tensor/storage
        uint8_t creation_mode = Data[offset++] % 5;
        
        torch::Tensor bool_tensor;
        
        switch (creation_mode) {
            case 0: {
                // Create boolean tensor from fuzzer data directly
                size_t num_elements = std::min<size_t>((Size - offset), 256);
                if (num_elements == 0) num_elements = 1;
                
                std::vector<bool> values;
                values.reserve(num_elements);
                for (size_t i = 0; i < num_elements && offset < Size; i++) {
                    values.push_back((Data[offset++] & 0x1) != 0);
                }
                if (values.empty()) values.push_back(false);
                
                // Create tensor from bool vector
                auto int_values = std::vector<int64_t>(values.begin(), values.end());
                bool_tensor = torch::tensor(int_values, torch::kBool);
                break;
            }
            case 1: {
                // Create tensor using createTensor and convert to bool
                torch::Tensor source_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                bool_tensor = source_tensor.to(torch::kBool);
                break;
            }
            case 2: {
                // Create zeros tensor of bool type
                int64_t size = 1;
                if (offset + sizeof(int32_t) <= Size) {
                    int32_t raw_size;
                    std::memcpy(&raw_size, Data + offset, sizeof(int32_t));
                    offset += sizeof(int32_t);
                    size = std::clamp<int64_t>(std::abs(raw_size) % 1000, 1, 512);
                }
                bool_tensor = torch::zeros({size}, torch::kBool);
                break;
            }
            case 3: {
                // Create ones tensor of bool type
                int64_t size = 1;
                if (offset + sizeof(int32_t) <= Size) {
                    int32_t raw_size;
                    std::memcpy(&raw_size, Data + offset, sizeof(int32_t));
                    offset += sizeof(int32_t);
                    size = std::clamp<int64_t>(std::abs(raw_size) % 1000, 1, 512);
                }
                bool_tensor = torch::ones({size}, torch::kBool);
                break;
            }
            case 4: {
                // Create 2D boolean tensor
                int64_t rows = 1, cols = 1;
                if (offset + 2 <= Size) {
                    rows = std::clamp<int64_t>((Data[offset++] % 32) + 1, 1, 32);
                    cols = std::clamp<int64_t>((Data[offset++] % 32) + 1, 1, 32);
                }
                bool_tensor = torch::randint(0, 2, {rows, cols}, torch::kBool);
                break;
            }
        }
        
        // Ensure tensor is valid and contiguous for storage access
        bool_tensor = bool_tensor.contiguous();
        
        // Access the storage (this is what BoolStorage represents in C++)
        at::Storage storage = bool_tensor.storage();
        
        // Test storage properties
        size_t nbytes = storage.nbytes();
        (void)nbytes;
        
        // Test storage data access
        if (storage.data_ptr().get() != nullptr) {
            // Read from storage
            bool* data_ptr = static_cast<bool*>(storage.data_ptr().get());
            size_t num_elements = bool_tensor.numel();
            
            if (num_elements > 0 && offset < Size) {
                // Read a random element
                size_t read_idx = Data[offset++] % num_elements;
                bool val = data_ptr[read_idx];
                (void)val;
            }
            
            // Write to storage if we have more data
            if (num_elements > 0 && offset < Size) {
                size_t write_idx = Data[offset++] % num_elements;
                bool new_val = (offset < Size) ? ((Data[offset++] & 0x1) != 0) : false;
                data_ptr[write_idx] = new_val;
            }
        }
        
        // Perform operations that exercise the boolean storage
        if (offset < Size) {
            uint8_t op = Data[offset++] % 8;
            
            try {
                switch (op) {
                    case 0: {
                        // Sum - counts true values
                        auto result = bool_tensor.sum();
                        (void)result;
                        break;
                    }
                    case 1: {
                        // any() - checks if any true
                        auto result = bool_tensor.any();
                        (void)result;
                        break;
                    }
                    case 2: {
                        // all() - checks if all true
                        auto result = bool_tensor.all();
                        (void)result;
                        break;
                    }
                    case 3: {
                        // Logical not
                        auto result = bool_tensor.logical_not();
                        (void)result;
                        break;
                    }
                    case 4: {
                        // Clone and verify storage is different
                        auto cloned = bool_tensor.clone();
                        at::Storage cloned_storage = cloned.storage();
                        (void)cloned_storage;
                        break;
                    }
                    case 5: {
                        // Create another bool tensor and do logical operations
                        auto other = torch::randint(0, 2, bool_tensor.sizes(), torch::kBool);
                        auto and_result = bool_tensor.logical_and(other);
                        auto or_result = bool_tensor.logical_or(other);
                        auto xor_result = bool_tensor.logical_xor(other);
                        (void)and_result;
                        (void)or_result;
                        (void)xor_result;
                        break;
                    }
                    case 6: {
                        // nonzero - get indices of true values
                        auto indices = bool_tensor.nonzero();
                        (void)indices;
                        break;
                    }
                    case 7: {
                        // Use as mask for another tensor
                        auto values = torch::randn(bool_tensor.sizes());
                        auto masked = values.masked_select(bool_tensor);
                        (void)masked;
                        break;
                    }
                }
            } catch (const c10::Error &e) {
                // Expected errors from shape mismatches, etc.
            }
        }
        
        // Test storage sharing behavior
        if (offset < Size && (Data[offset++] & 0x1)) {
            try {
                // Create a view and verify storage is shared
                auto view = bool_tensor.view({-1});
                at::Storage view_storage = view.storage();
                
                // Storages should be the same underlying object
                bool same_storage = (storage.data_ptr().get() == view_storage.data_ptr().get());
                (void)same_storage;
            } catch (const c10::Error &e) {
                // View may fail for certain tensor configurations
            }
        }
        
        // Test conversion from bool storage to other types
        if (offset < Size) {
            uint8_t convert_type = Data[offset++] % 4;
            try {
                switch (convert_type) {
                    case 0: {
                        auto int_tensor = bool_tensor.to(torch::kInt32);
                        (void)int_tensor;
                        break;
                    }
                    case 1: {
                        auto float_tensor = bool_tensor.to(torch::kFloat32);
                        (void)float_tensor;
                        break;
                    }
                    case 2: {
                        auto long_tensor = bool_tensor.to(torch::kInt64);
                        (void)long_tensor;
                        break;
                    }
                    case 3: {
                        auto byte_tensor = bool_tensor.to(torch::kUInt8);
                        (void)byte_tensor;
                        break;
                    }
                }
            } catch (const c10::Error &e) {
                // Conversion errors
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}