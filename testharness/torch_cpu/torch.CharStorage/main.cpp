#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor to get some data for the CharStorage
        torch::Tensor tensor;
        if (offset < Size) {
            tensor = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Get size for CharStorage
        int64_t storage_size = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&storage_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure storage_size is reasonable
            storage_size = std::abs(storage_size) % 1000 + 1;
        }
        
        // Create CharStorage with different constructors
        try {
            // Default constructor
            c10::Storage storage1 = c10::make_intrusive<c10::StorageImpl>(
                c10::StorageImpl::use_byte_size_t(),
                0,
                c10::GetDefaultCPUAllocator(),
                true);
            
            // Constructor with size
            c10::Storage storage2 = c10::make_intrusive<c10::StorageImpl>(
                c10::StorageImpl::use_byte_size_t(),
                storage_size,
                c10::GetDefaultCPUAllocator(),
                true);
            
            // Fill with some values
            for (int64_t i = 0; i < storage_size && i < tensor.numel(); i++) {
                int8_t value = 0;
                if (i < tensor.numel()) {
                    // Extract a value from the tensor
                    if (tensor.dtype() == torch::kInt8) {
                        value = tensor.data_ptr<int8_t>()[i % tensor.numel()];
                    } else if (tensor.dtype() == torch::kByte || tensor.dtype() == torch::kUInt8) {
                        value = static_cast<int8_t>(tensor.data_ptr<uint8_t>()[i % tensor.numel()]);
                    } else {
                        // Just use a value derived from the tensor's data pointer
                        value = static_cast<int8_t>(reinterpret_cast<uintptr_t>(tensor.data_ptr()) & 0xFF);
                    }
                } else if (offset < Size) {
                    // Use data from the input
                    value = static_cast<int8_t>(Data[offset++]);
                }
                
                if (storage2.data_ptr().get()) {
                    static_cast<char*>(storage2.data_ptr().get())[i] = value;
                }
            }
            
            // Copy constructor
            c10::Storage storage3 = storage2;
            
            // Move constructor
            c10::Storage storage4 = std::move(storage3);
            
            // Test other operations
            if (storage_size > 0) {
                // Test data() access
                char* data_ptr = static_cast<char*>(storage2.data_ptr().get());
                if (data_ptr && offset < Size) {
                    data_ptr[0] = static_cast<char>(Data[offset++]);
                }
                
                // Test size() and resize()
                int64_t new_size = storage_size / 2 + 1;
                storage2.unsafeGetStorageImpl()->set_nbytes(new_size);
                
                // Test copy_
                if (storage4.nbytes() > 0 && storage2.nbytes() > 0) {
                    if (storage2.data_ptr().get() && storage4.data_ptr().get()) {
                        std::memcpy(storage2.data_ptr().get(), storage4.data_ptr().get(), 
                                  std::min(storage2.nbytes(), storage4.nbytes()));
                    }
                }
            }
            
            // Test empty storage
            c10::Storage empty_storage = c10::make_intrusive<c10::StorageImpl>(
                c10::StorageImpl::use_byte_size_t(),
                0,
                c10::GetDefaultCPUAllocator(),
                true);
            
            // Test from blob if we have data
            if (offset < Size) {
                size_t blob_size = std::min<size_t>(Size - offset, 100);
                c10::Storage from_blob = c10::make_intrusive<c10::StorageImpl>(
                    c10::StorageImpl::use_byte_size_t(),
                    blob_size,
                    const_cast<void*>(static_cast<const void*>(Data + offset)),
                    c10::GetDefaultCPUAllocator(),
                    false);
                offset += blob_size;
            }
            
            // Test with negative size (should throw)
            try {
                c10::Storage negative_size = c10::make_intrusive<c10::StorageImpl>(
                    c10::StorageImpl::use_byte_size_t(),
                    -1,
                    c10::GetDefaultCPUAllocator(),
                    true);
            } catch (const c10::Error& e) {
                // Expected exception
            }
            
            // Test with very large size (should throw or handle gracefully)
            try {
                int64_t large_size = std::numeric_limits<int64_t>::max();
                c10::Storage large_storage = c10::make_intrusive<c10::StorageImpl>(
                    c10::StorageImpl::use_byte_size_t(),
                    large_size,
                    c10::GetDefaultCPUAllocator(),
                    true);
            } catch (const std::exception& e) {
                // Expected exception
            }
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected in some cases
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}