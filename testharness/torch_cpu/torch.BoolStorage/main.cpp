#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a meaningful test
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to get data for the BoolStorage
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a BoolStorage
        try {
            // Create a Storage for bool type
            at::Storage storage;
            
            // Try different ways to initialize the Storage
            if (offset < Size) {
                uint8_t option = Data[offset++] % 4;
                
                switch (option) {
                    case 0: {
                        // Default constructor already called
                        break;
                    }
                    case 1: {
                        // Create with size
                        int64_t size = 1;
                        if (offset + sizeof(int64_t) <= Size) {
                            std::memcpy(&size, Data + offset, sizeof(int64_t));
                            offset += sizeof(int64_t);
                        }
                        if (size > 0) {
                            storage = at::Storage(at::Storage::use_byte_size_t(), size * sizeof(bool), at::DataPtr(nullptr, at::Device(at::kCPU)), nullptr, false);
                        }
                        break;
                    }
                    case 2: {
                        // Create from tensor data if tensor is boolean type
                        if (tensor.dtype() == torch::kBool) {
                            // Convert tensor to contiguous if needed
                            auto contiguous_tensor = tensor.contiguous();
                            storage = contiguous_tensor.storage();
                        } else {
                            // Create a new boolean tensor and use its data
                            auto bool_tensor = tensor.to(torch::kBool);
                            auto contiguous_bool = bool_tensor.contiguous();
                            storage = contiguous_bool.storage();
                        }
                        break;
                    }
                    case 3: {
                        // Create from vector of bools
                        std::vector<bool> values;
                        size_t num_values = (Size - offset > 100) ? 100 : (Size - offset);
                        for (size_t i = 0; i < num_values; i++) {
                            values.push_back(Data[offset + i] & 0x1); // Convert to bool
                        }
                        offset += num_values;
                        if (!values.empty()) {
                            auto bool_tensor = torch::tensor(values, torch::kBool);
                            storage = bool_tensor.storage();
                        }
                        break;
                    }
                }
            }
            
            // Test Storage operations
            if (offset < Size) {
                uint8_t op = Data[offset++] % 5;
                
                switch (op) {
                    case 0: {
                        // Get size
                        if (storage.defined()) {
                            auto size = storage.nbytes();
                        }
                        break;
                    }
                    case 1: {
                        // Access elements if storage is not empty
                        if (storage.defined() && storage.nbytes() > 0) {
                            size_t idx = 0;
                            if (offset + sizeof(size_t) <= Size) {
                                std::memcpy(&idx, Data + offset, sizeof(size_t));
                                offset += sizeof(size_t);
                            }
                            size_t max_elements = storage.nbytes() / sizeof(bool);
                            if (max_elements > 0) {
                                idx = idx % max_elements; // Ensure valid index
                                if (storage.data_ptr().get()) {
                                    bool* data = static_cast<bool*>(storage.data_ptr().get());
                                    bool val = data[idx];
                                }
                            }
                        }
                        break;
                    }
                    case 2: {
                        // Resize storage
                        int64_t new_size = 1;
                        if (offset + sizeof(int64_t) <= Size) {
                            std::memcpy(&new_size, Data + offset, sizeof(int64_t));
                            offset += sizeof(int64_t);
                            if (new_size > 0 && storage.defined()) {
                                storage.resize_(new_size * sizeof(bool));
                            }
                        }
                        break;
                    }
                    case 3: {
                        // Fill with value
                        bool fill_value = false;
                        if (offset < Size) {
                            fill_value = (Data[offset++] & 0x1);
                        }
                        if (storage.defined() && storage.data_ptr().get()) {
                            size_t num_elements = storage.nbytes() / sizeof(bool);
                            bool* data = static_cast<bool*>(storage.data_ptr().get());
                            for (size_t i = 0; i < num_elements; i++) {
                                data[i] = fill_value;
                            }
                        }
                        break;
                    }
                    case 4: {
                        // Copy from another storage
                        int64_t other_size = 10;
                        if (offset + sizeof(int64_t) <= Size) {
                            std::memcpy(&other_size, Data + offset, sizeof(int64_t));
                            offset += sizeof(int64_t);
                            if (other_size > 0) {
                                std::vector<bool> other_values;
                                for (int64_t i = 0; i < other_size; i++) {
                                    if (offset < Size) {
                                        other_values.push_back(Data[offset++] & 0x1);
                                    }
                                }
                                auto other_tensor = torch::tensor(other_values, torch::kBool);
                                auto other_storage = other_tensor.storage();
                                
                                // Copy to original storage
                                if (storage.defined() && other_storage.defined()) {
                                    size_t copy_size = std::min(storage.nbytes(), other_storage.nbytes());
                                    if (storage.data_ptr().get() && other_storage.data_ptr().get()) {
                                        std::memcpy(storage.data_ptr().get(), other_storage.data_ptr().get(), copy_size);
                                    }
                                }
                            }
                        }
                        break;
                    }
                }
            }
            
            // Create a tensor from the storage
            torch::Tensor storage_tensor;
            if (storage.defined() && storage.nbytes() > 0) {
                size_t num_elements = storage.nbytes() / sizeof(bool);
                storage_tensor = torch::from_blob(storage.data_ptr().get(), {static_cast<int64_t>(num_elements)}, torch::kBool);
            }
        } catch (const c10::Error &e) {
            // PyTorch specific errors are expected and part of testing
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
