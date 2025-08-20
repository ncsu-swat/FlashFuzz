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
        
        // Get the tensor's storage
        auto storage = tensor.storage();
        
        // Test storage operations directly
        size_t size = storage.nbytes();
        auto device = storage.device();
        
        // Test data access
        if (size > 0) {
            void* data_ptr = storage.data();
        }
        
        // Test copy operations
        auto copy = storage.clone();
        
        // Test resizing if there's enough data left
        if (offset + 1 < Size) {
            uint8_t resize_value = Data[offset++];
            size_t new_size = resize_value % 32;  // Keep size reasonable
            storage.resize(new_size);
        }
        
        // Test creating a tensor from the storage
        auto options = torch::TensorOptions().dtype(tensor.dtype()).device(tensor.device());
        std::vector<int64_t> sizes;
        
        // Create a simple shape based on the tensor size
        if (tensor.numel() > 0) {
            if (offset < Size) {
                uint8_t dim_count = Data[offset++] % 4 + 1;  // 1-4 dimensions
                sizes.reserve(dim_count);
                
                int64_t remaining_elements = tensor.numel();
                for (uint8_t i = 0; i < dim_count - 1 && remaining_elements > 1; ++i) {
                    int64_t dim_size = 1;
                    if (offset < Size) {
                        dim_size = (Data[offset++] % 8) + 1;  // 1-8 size
                        dim_size = std::min(dim_size, remaining_elements);
                    }
                    sizes.push_back(dim_size);
                    remaining_elements /= dim_size;
                }
                sizes.push_back(std::max(int64_t(1), remaining_elements));
            } else {
                // Default to a 1D tensor
                sizes.push_back(tensor.numel());
            }
            
            // Create tensor from storage
            torch::Tensor tensor_from_storage = torch::empty(sizes, options);
            tensor_from_storage.set_(storage);
        }
        
        // Test moving the storage to a different device if CUDA is available
        #ifdef USE_CUDA
        if (torch::cuda::is_available()) {
            auto cuda_storage = storage.to(torch::kCUDA);
            auto cpu_storage = cuda_storage.to(torch::kCPU);
        }
        #endif
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}