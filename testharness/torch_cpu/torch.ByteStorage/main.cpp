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
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get the number of elements in the tensor
        int64_t numel = tensor.numel();
        
        // Create a ByteStorage by creating a tensor first
        torch::Tensor byte_tensor = torch::empty({numel}, torch::kByte);
        torch::Storage storage = byte_tensor.storage();
        
        // Try different operations with ByteStorage
        
        // 1. Create a ByteStorage from tensor data
        if (tensor.dtype() == torch::kByte || tensor.dtype() == torch::kUInt8) {
            // Direct copy if tensor is already byte type
            storage = tensor.storage();
        } else {
            // Convert tensor to byte type first
            torch::Tensor converted_tensor = tensor.to(torch::kByte);
            storage = converted_tensor.storage();
        }
        
        // 2. Test storage size
        int64_t storage_size = storage.nbytes();
        
        // 3. Test storage data access
        if (storage_size > 0) {
            const void* data_ptr = storage.data();
            const uint8_t* byte_data = static_cast<const uint8_t*>(data_ptr);
            uint8_t first_byte = byte_data[0];
            
            // Test mutable data access
            void* mutable_data_ptr = storage.mutable_data();
            uint8_t* mutable_byte_data = static_cast<uint8_t*>(mutable_data_ptr);
            
            // Test setting values
            mutable_byte_data[0] = 255;
            
            // Test setting values at the end
            if (storage_size > 1) {
                mutable_byte_data[storage_size - 1] = 128;
            }
        }
        
        // 4. Test storage resize
        if (offset + sizeof(int64_t) <= Size) {
            int64_t new_size;
            std::memcpy(&new_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make sure new_size is reasonable
            new_size = std::abs(new_size) % 1000;
            
            // Resize by creating a new tensor
            torch::Tensor resized_tensor = torch::empty({new_size}, torch::kByte);
            storage = resized_tensor.storage();
            storage_size = storage.nbytes();
        }
        
        // 5. Test storage fill by creating tensor and filling it
        if (offset < Size) {
            uint8_t fill_value = Data[offset++];
            torch::Tensor fill_tensor = torch::from_blob(storage.mutable_data(), {storage_size}, torch::kByte);
            fill_tensor.fill_(fill_value);
        }
        
        // 6. Test creating a tensor from storage
        torch::Tensor new_tensor = torch::from_blob(storage.mutable_data(), {storage_size}, torch::kByte);
        
        // 7. Test storage copy by creating new tensor
        torch::Tensor copy_tensor = torch::empty({storage_size}, torch::kByte);
        torch::Storage storage_copy = copy_tensor.storage();
        if (storage_size > 0) {
            std::memcpy(storage_copy.mutable_data(), storage.data(), storage_size);
        }
        
        // 8. Test storage view by creating a slice
        if (storage_size > 2 && offset + sizeof(int64_t) <= Size) {
            int64_t offset_val;
            std::memcpy(&offset_val, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make sure offset is within bounds
            offset_val = std::abs(offset_val) % (storage_size - 1);
            
            int64_t size_val = storage_size - offset_val;
            torch::Tensor view_tensor = torch::from_blob(
                static_cast<uint8_t*>(storage.mutable_data()) + offset_val, 
                {size_val}, 
                torch::kByte
            );
            torch::Storage storage_view = view_tensor.storage();
        }
        
        // 9. Test storage to CPU (create tensor on CPU)
        torch::Tensor cpu_tensor = torch::from_blob(storage.mutable_data(), {storage_size}, torch::kByte).cpu();
        torch::Storage cpu_storage = cpu_tensor.storage();
        
        // 10. Test storage data pointer access
        const void* const_data_ptr = storage.data();
        void* mutable_data_ptr = storage.mutable_data();
        
        // 11. Test storage element size (byte storage has element size 1)
        size_t element_size = 1; // ByteStorage always has element size 1
        
        // 12. Test storage device
        torch::Device device = storage.device();
        
        // 13. Test storage data type (create tensor to check dtype)
        torch::Tensor dtype_tensor = torch::from_blob(storage.mutable_data(), {storage_size}, torch::kByte);
        torch::ScalarType dtype = dtype_tensor.dtype().toScalarType();
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
