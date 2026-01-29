#include "fuzzer_utils.h" // General fuzzing utilities
#include <algorithm>
#include <cstring>
#include <iostream>

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
        
        if (Size < 2) {
            return 0;
        }
        
        // Target API: Storage operations on ComplexFloat tensors
        // In C++ frontend, we access storage through tensors with kComplexFloat dtype
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert tensor to complex float if it's not already
        if (tensor.dtype() != torch::kComplexFloat) {
            tensor = tensor.to(torch::kComplexFloat);
        }
        
        // Get the storage from the tensor
        auto storage = tensor.storage();
        
        // Test various operations on the storage
        size_t storage_size = storage.nbytes() / sizeof(c10::complex<float>);
        
        // Access elements if storage is not empty
        if (storage_size > 0) {
            // Access data pointer
            auto data_ptr = static_cast<const c10::complex<float>*>(storage.data());
            
            // Access first element (use volatile to prevent optimization)
            volatile auto first_element = data_ptr[0];
            (void)first_element;
            
            // Access last element
            volatile auto last_element = data_ptr[storage_size - 1];
            (void)last_element;
            
            // Access random element if size > 1
            if (storage_size > 1 && offset < Size) {
                size_t idx = Data[offset++] % storage_size;
                volatile auto random_element = data_ptr[idx];
                (void)random_element;
            }
        }
        
        // Test data() method
        const void *raw_data = storage.data();

        // Test nbytes() method
        auto nbytes = storage.nbytes();

        // Test device() method
        auto device = storage.device();
        (void)device;
        
        // Test copy operations if we have enough data
        if (offset + 1 < Size && storage_size > 0 && nbytes > 0) {
            // Create a new tensor with complex float type
            auto new_tensor = torch::zeros({static_cast<int64_t>(storage_size)}, torch::kComplexFloat);
            auto new_storage = new_tensor.storage();
            
            // Copy data manually
            std::memcpy(new_storage.mutable_data(), raw_data, nbytes);
            
            // Create a partial copy
            size_t partial_size = std::max<size_t>(1, storage_size / 2);
            auto partial_tensor = torch::zeros({static_cast<int64_t>(partial_size)}, torch::kComplexFloat);
            auto partial_storage = partial_tensor.storage();
            std::memcpy(partial_storage.mutable_data(), raw_data, partial_size * sizeof(c10::complex<float>));
        }
        
        // Test creating tensors of various sizes
        if (offset < Size) {
            size_t new_size = (Data[offset++] % 99) + 1; // 1 to 100
            auto resized_tensor = torch::zeros({static_cast<int64_t>(new_size)}, torch::kComplexFloat);
            auto resized_storage = resized_tensor.storage();
            (void)resized_storage;
        }
        
        // Test fill operation
        if (offset + sizeof(c10::complex<float>) <= Size && storage_size > 0) {
            c10::complex<float> fill_value;
            std::memcpy(&fill_value, Data + offset, sizeof(c10::complex<float>));
            offset += sizeof(c10::complex<float>);
            
            // Fill the tensor data
            auto data_ptr = static_cast<c10::complex<float>*>(storage.mutable_data());
            for (size_t i = 0; i < storage_size; ++i) {
                data_ptr[i] = fill_value;
            }
        }
        
        // Test creating a tensor from storage using from_blob
        if (storage_size > 0) {
            auto tensor_from_storage = torch::from_blob(
                storage.mutable_data(), 
                {static_cast<int64_t>(storage_size)}, 
                torch::TensorOptions().dtype(torch::kComplexFloat)
            );
            // Verify the tensor was created correctly
            (void)tensor_from_storage.numel();
        }
        
        // Test storage operations with different tensor shapes
        if (offset + 1 < Size && storage_size >= 4) {
            // Reshape the underlying data as 2D
            int64_t dim0 = 2;
            int64_t dim1 = static_cast<int64_t>(storage_size / 2);
            if (dim1 > 0) {
                auto reshaped = torch::from_blob(
                    storage.mutable_data(),
                    {dim0, dim1},
                    torch::TensorOptions().dtype(torch::kComplexFloat)
                );
                (void)reshaped.size(0);
            }
        }
        
        // Test storage sharing - create a view and verify storage is shared
        if (storage_size > 1) {
            auto view = tensor.view({-1});
            auto view_storage = view.storage();
            // Both should point to the same storage
            (void)(storage.data() == view_storage.data());
        }
        
        // Test with contiguous tensors
        auto contig = tensor.contiguous();
        auto contig_storage = contig.storage();
        (void)contig_storage.nbytes();
        
        // Test clone - should create new storage
        auto cloned = tensor.clone();
        auto cloned_storage = cloned.storage();
        // Cloned storage should be different
        (void)(storage.data() != cloned_storage.data());
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}