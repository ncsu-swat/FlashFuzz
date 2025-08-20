#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get the tensor's storage as Storage
        at::Storage storage = tensor.storage();
        
        // Test various Storage operations
        
        // Get storage size
        size_t storage_size = storage.nbytes();
        
        // Get storage data pointer
        void* data_ptr = storage.data();
        
        // Get storage device
        torch::Device device = storage.device();
        
        // Test storage dtype
        auto dtype = storage.dtype();
        
        // Test storage nbytes
        size_t nbytes = storage.nbytes();
        
        // Test storage clone
        at::Storage cloned_storage = storage.clone();
        
        // Test storage copy_
        if (offset + 1 < Size) {
            // Create another tensor to get a different storage
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            at::Storage another_storage = another_tensor.storage();
            
            // Try to copy if sizes match
            if (storage.nbytes() == another_storage.nbytes() && 
                storage.dtype() == another_storage.dtype() &&
                storage.device().type() == another_storage.device().type()) {
                storage.copy_(another_storage);
            }
        }
        
        // Test storage resize
        if (offset < Size) {
            size_t new_size = Data[offset] % 100;
            storage.resize_(new_size);
        }
        
        // Test storage set_data_ptr
        if (storage.nbytes() > 0) {
            // Allocate memory for new data
            void* new_data = malloc(storage.nbytes());
            if (new_data) {
                // Fill with some data
                memset(new_data, 0x42, storage.nbytes());
                
                // Create a deleter function
                auto deleter = [](void* ptr) { free(ptr); };
                
                // Set the data pointer
                storage.set_data_ptr(at::DataPtr(new_data, new_data, deleter, device));
            }
        }
        
        // Test Storage constructor with size and options
        if (offset < Size) {
            size_t size = Data[offset] % 100;
            c10::TensorOptions options = c10::TensorOptions().dtype(fuzzer_utils::parseDataType(Data[offset]));
            at::Storage new_storage(c10::Storage::create_legacy(device, options.dtype().toScalarType()));
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}