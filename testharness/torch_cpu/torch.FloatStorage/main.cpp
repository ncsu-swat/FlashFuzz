#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create meaningful input
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a Storage from the tensor
        at::Storage storage;
        
        // Try different ways to create/use Storage based on available bytes
        if (offset < Size) {
            uint8_t storage_op = Data[offset++] % 4;
            
            switch (storage_op) {
                case 0: {
                    // Create empty storage
                    storage = at::Storage();
                    break;
                }
                case 1: {
                    // Create storage with size
                    int64_t size = 0;
                    if (offset + sizeof(int64_t) <= Size) {
                        std::memcpy(&size, Data + offset, sizeof(int64_t));
                        offset += sizeof(int64_t);
                    }
                    // Ensure positive size
                    if (size > 0 && size < 1000000) {
                        storage = at::Storage(at::StorageImpl::create(
                            c10::scalarTypeToTypeMeta(torch::kFloat),
                            size,
                            c10::GetDefaultCPUAllocator(),
                            true
                        ));
                    }
                    break;
                }
                case 2: {
                    // Create storage from tensor
                    if (tensor.dtype() == torch::kFloat) {
                        storage = tensor.storage();
                    } else {
                        // Convert tensor to float if needed
                        auto float_tensor = tensor.to(torch::kFloat);
                        storage = float_tensor.storage();
                    }
                    break;
                }
                case 3: {
                    // Create storage and fill with value
                    int64_t size = 10; // Default size
                    float fill_value = 0.0f;
                    
                    if (offset + sizeof(int64_t) <= Size) {
                        std::memcpy(&size, Data + offset, sizeof(int64_t));
                        offset += sizeof(int64_t);
                    }
                    
                    if (offset + sizeof(float) <= Size) {
                        std::memcpy(&fill_value, Data + offset, sizeof(float));
                        offset += sizeof(float);
                    }
                    
                    // Ensure positive size
                    if (size > 0 && size < 1000000) {
                        storage = at::Storage(at::StorageImpl::create(
                            c10::scalarTypeToTypeMeta(torch::kFloat),
                            size,
                            c10::GetDefaultCPUAllocator(),
                            true
                        ));
                        
                        // Fill with value
                        float* data_ptr = static_cast<float*>(storage.data());
                        for (int64_t i = 0; i < size; ++i) {
                            data_ptr[i] = fill_value;
                        }
                    }
                    break;
                }
            }
        } else {
            // Default case if we don't have enough bytes
            storage = at::Storage();
        }
        
        // Test storage operations if we have a valid storage
        if (storage.defined()) {
            // Get size
            auto size = storage.nbytes() / sizeof(float);
            
            // Test element access if storage has elements
            if (size > 0) {
                float* data_ptr = static_cast<float*>(storage.data());
                
                // Access first element
                float first_element = data_ptr[0];
                
                // Access last element
                float last_element = data_ptr[size - 1];
                
                // Try to modify an element
                if (offset < Size) {
                    float new_value = 0.0f;
                    if (offset + sizeof(float) <= Size) {
                        std::memcpy(&new_value, Data + offset, sizeof(float));
                        offset += sizeof(float);
                    }
                    
                    size_t index = 0;
                    if (offset + sizeof(size_t) <= Size) {
                        std::memcpy(&index, Data + offset, sizeof(size_t));
                        offset += sizeof(size_t);
                        
                        // Ensure valid index
                        if (index < size) {
                            data_ptr[index] = new_value;
                        }
                    } else {
                        data_ptr[0] = new_value;
                    }
                }
            }
            
            // Test data() method
            void* data_ptr = storage.data();
            
            // Test copy constructor
            at::Storage storage_copy(storage);
            
            // Test move constructor
            at::Storage storage_moved(std::move(storage_copy));
            
            // Test assignment operator
            at::Storage another_storage;
            another_storage = storage_moved;
            
            // Test is_pinned
            bool is_pinned = storage.is_pinned();
            
            // Test nbytes
            size_t nbytes = storage.nbytes();
            
            // Test device
            auto device = storage.device();
            
            // Test dtype
            auto dtype = storage.dtype();
            
            // Test to method
            if (offset < Size) {
                uint8_t device_type = Data[offset++] % 2;
                torch::Device target_device(device_type == 0 ? torch::kCPU : torch::kCUDA);
                
                // Only attempt CUDA if available
                if (target_device.is_cuda() && !torch::cuda::is_available()) {
                    target_device = torch::Device(torch::kCPU);
                }
                
                auto moved_storage = storage.to(target_device);
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