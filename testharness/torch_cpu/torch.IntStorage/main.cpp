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
        
        // Need at least a few bytes to create meaningful test cases
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to get data for the Storage
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create Storage with different approaches
        
        // 1. Create empty Storage
        torch::Storage storage1;
        
        // 2. Create Storage with size
        int64_t size = 0;
        if (offset < Size) {
            std::memcpy(&size, Data + offset, std::min(sizeof(int64_t), Size - offset));
            offset += std::min(sizeof(int64_t), Size - offset);
            
            // Make size reasonable but allow edge cases
            size = std::abs(size) % 1000;
            
            torch::Storage storage2 = torch::Storage(c10::make_intrusive<c10::StorageImpl>(
                c10::StorageImpl::use_byte_size_t(),
                size * sizeof(int32_t),
                c10::GetDefaultCPUAllocator(),
                true));
            
            // Fill storage with some values
            if (storage2.data_ptr().get() != nullptr) {
                int32_t* data_ptr = static_cast<int32_t*>(storage2.data_ptr().get());
                for (int64_t i = 0; i < size && offset < Size; i++) {
                    int32_t value = 0;
                    std::memcpy(&value, Data + offset, std::min(sizeof(int32_t), Size - offset));
                    offset += std::min(sizeof(int32_t), Size - offset);
                    data_ptr[i] = value;
                }
            }
        }
        
        // 3. Create Storage from tensor data if tensor has integer type
        if (tensor.dtype() == torch::kInt32 || tensor.dtype() == torch::kInt64) {
            // Convert tensor to contiguous if needed
            auto contiguousTensor = tensor.contiguous();
            
            // Get the underlying storage
            torch::Storage storage3 = contiguousTensor.storage();
            
            // Test storage operations
            if (storage3.nbytes() > 0) {
                // Access elements
                if (tensor.numel() > 0) {
                    int64_t idx = std::abs(tensor[0].item<int64_t>()) % tensor.numel();
                    
                    // Resize storage
                    if (offset < Size) {
                        int64_t newSize = 0;
                        std::memcpy(&newSize, Data + offset, std::min(sizeof(int64_t), Size - offset));
                        offset += std::min(sizeof(int64_t), Size - offset);
                        
                        // Make size reasonable but allow edge cases
                        newSize = std::abs(newSize) % 1000;
                        
                        storage3.resize_(newSize * tensor.element_size());
                    }
                }
            }
        }
        
        // 4. Create Storage from data pointer
        std::vector<int32_t> data;
        int32_t dataSize = 10;
        if (offset < Size) {
            std::memcpy(&dataSize, Data + offset, std::min(sizeof(int32_t), Size - offset));
            offset += std::min(sizeof(int32_t), Size - offset);
            
            // Make size reasonable but allow edge cases
            dataSize = std::abs(dataSize) % 100;
            
            // Prepare data
            data.resize(dataSize);
            for (int i = 0; i < dataSize && offset < Size; i++) {
                std::memcpy(&data[i], Data + offset, std::min(sizeof(int32_t), Size - offset));
                offset += std::min(sizeof(int32_t), Size - offset);
            }
            
            // Create storage from data
            torch::Storage storage4 = torch::Storage(c10::make_intrusive<c10::StorageImpl>(
                c10::StorageImpl::use_byte_size_t(),
                data.size() * sizeof(int32_t),
                c10::GetDefaultCPUAllocator(),
                true));
            
            if (storage4.data_ptr().get() != nullptr) {
                std::memcpy(storage4.data_ptr().get(), data.data(), data.size() * sizeof(int32_t));
            }
            
            // Test copy constructor
            torch::Storage storage5 = storage4;
        }
        
        // 5. Test with empty/zero-sized storage
        torch::Storage emptyStorage = torch::Storage(c10::make_intrusive<c10::StorageImpl>(
            c10::StorageImpl::use_byte_size_t(),
            0,
            c10::GetDefaultCPUAllocator(),
            true));
        
        // 6. Test with large size (should work but be careful with memory)
        if (offset < Size) {
            int64_t testSize = 0;
            std::memcpy(&testSize, Data + offset, std::min(sizeof(int64_t), Size - offset));
            offset += std::min(sizeof(int64_t), Size - offset);
            
            // Make size reasonable
            testSize = std::abs(testSize) % 1000 + 1;
            
            try {
                torch::Storage testStorage = torch::Storage(c10::make_intrusive<c10::StorageImpl>(
                    c10::StorageImpl::use_byte_size_t(),
                    testSize * sizeof(int32_t),
                    c10::GetDefaultCPUAllocator(),
                    true));
            } catch (const c10::Error& e) {
                // Expected exception for large sizes
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
