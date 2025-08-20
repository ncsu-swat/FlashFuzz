#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a meaningful test
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor with half precision data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a HalfStorage from the tensor
        try {
            // Convert tensor to Half if it's not already
            if (tensor.dtype() != torch::kHalf) {
                tensor = tensor.to(torch::kHalf);
            }
            
            // Get the storage from the tensor
            auto storage = tensor.storage();
            
            // Create a new storage for half precision
            auto halfStorage = c10::make_intrusive<at::StorageImpl>(
                c10::StorageImpl::use_byte_size_t(),
                storage.nbytes(),
                at::DataPtr(nullptr, at::Device(at::kCPU)),
                nullptr,
                false
            );
            
            // Copy data from tensor storage to HalfStorage
            std::memcpy(halfStorage->data(), storage.data_ptr(), storage.nbytes());
            
            // Test HalfStorage functionality
            size_t size = halfStorage->nbytes();
            void* data_ptr = halfStorage->data();
            
            // Create a new tensor from the HalfStorage
            auto options = torch::TensorOptions().dtype(torch::kHalf);
            auto new_tensor = torch::empty(tensor.sizes(), options);
            auto new_storage = new_tensor.storage();
            
            // Copy data from HalfStorage to the new tensor's storage
            std::memcpy(new_storage.data_ptr(), halfStorage->data(), halfStorage->nbytes());
            
            // Create a HalfStorage with specific size
            if (offset < Size) {
                uint32_t storage_size = 0;
                std::memcpy(&storage_size, Data + offset, std::min(Size - offset, sizeof(uint32_t)));
                offset += sizeof(uint32_t);
                
                // Limit size to avoid excessive memory usage
                storage_size = storage_size % 1000 + 1;
                
                auto sizedStorage = c10::make_intrusive<at::StorageImpl>(
                    c10::StorageImpl::use_byte_size_t(),
                    storage_size * sizeof(at::Half),
                    at::DataPtr(nullptr, at::Device(at::kCPU)),
                    nullptr,
                    false
                );
                
                // Test resize functionality
                if (offset < Size) {
                    uint32_t new_size = 0;
                    std::memcpy(&new_size, Data + offset, std::min(Size - offset, sizeof(uint32_t)));
                    offset += sizeof(uint32_t);
                    
                    // Limit size to avoid excessive memory usage
                    new_size = new_size % 1000 + 1;
                    
                    sizedStorage->resize_(new_size * sizeof(at::Half));
                }
                
                // Test fill functionality
                if (offset < Size) {
                    uint16_t fill_value = 0;
                    std::memcpy(&fill_value, Data + offset, std::min(Size - offset, sizeof(uint16_t)));
                    offset += sizeof(uint16_t);
                    
                    at::Half half_val;
                    std::memcpy(&half_val, &fill_value, sizeof(uint16_t));
                    
                    at::Half* data_array = static_cast<at::Half*>(sizedStorage->data());
                    size_t num_elements = sizedStorage->nbytes() / sizeof(at::Half);
                    for (size_t i = 0; i < num_elements; ++i) {
                        data_array[i] = half_val;
                    }
                }
            }
            
            // Create a HalfStorage from data
            if (offset < Size) {
                std::vector<at::Half> data_vec;
                size_t remaining = Size - offset;
                size_t num_elements = remaining / sizeof(at::Half);
                
                // Limit size to avoid excessive memory usage
                num_elements = std::min(num_elements, static_cast<size_t>(1000));
                
                data_vec.resize(num_elements);
                std::memcpy(data_vec.data(), Data + offset, num_elements * sizeof(at::Half));
                
                auto dataStorage = c10::make_intrusive<at::StorageImpl>(
                    c10::StorageImpl::use_byte_size_t(),
                    num_elements * sizeof(at::Half),
                    at::DataPtr(nullptr, at::Device(at::kCPU)),
                    nullptr,
                    false
                );
                
                std::memcpy(dataStorage->data(), data_vec.data(), num_elements * sizeof(at::Half));
                
                // Test element access
                if (num_elements > 0) {
                    size_t idx = Data[offset % Size] % num_elements;
                    at::Half* data_array = static_cast<at::Half*>(dataStorage->data());
                    at::Half val = data_array[idx];
                    data_array[idx] = val;
                }
            }
        }
        catch (const std::exception& e) {
            // Catch exceptions from HalfStorage operations
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}