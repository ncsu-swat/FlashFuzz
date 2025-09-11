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
        
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the fuzzer data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Access tensor properties directly since torch::read_vitals doesn't exist
        auto memory_format = tensor.suggest_memory_format();
        auto storage_offset = tensor.storage_offset();
        auto sizes = tensor.sizes();
        auto strides = tensor.strides();
        
        // Try to create a new tensor with the same properties
        if (sizes.size() > 0 && strides.size() > 0) {
            auto options = torch::TensorOptions().dtype(tensor.dtype());
            auto new_tensor = torch::empty_strided(sizes, strides, options);
            
            // Copy data from original tensor to new tensor
            new_tensor.copy_(tensor);
        }
        
        // Try different variants of the API if there are any
        if (Size > offset && offset + 1 < Size) {
            uint8_t variant = Data[offset++];
            
            // Create another tensor for testing
            torch::Tensor tensor2;
            if (offset < Size) {
                tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                tensor2 = tensor.clone();
            }
            
            // Test with different tensors
            auto memory_format2 = tensor2.suggest_memory_format();
            auto storage_offset2 = tensor2.storage_offset();
            
            // Test with empty tensor
            torch::Tensor empty_tensor = torch::empty({0});
            auto empty_memory_format = empty_tensor.suggest_memory_format();
            auto empty_storage_offset = empty_tensor.storage_offset();
            
            // Test with scalar tensor
            torch::Tensor scalar_tensor = torch::tensor(3.14);
            auto scalar_memory_format = scalar_tensor.suggest_memory_format();
            auto scalar_storage_offset = scalar_tensor.storage_offset();
            
            // Test with boolean tensor
            torch::Tensor bool_tensor = torch::ones({2, 2}, torch::kBool);
            auto bool_memory_format = bool_tensor.suggest_memory_format();
            auto bool_storage_offset = bool_tensor.storage_offset();
            
            // Test with non-contiguous tensor
            if (tensor.dim() > 1 && tensor.size(0) > 1) {
                torch::Tensor non_contig = tensor.transpose(0, tensor.dim() - 1);
                auto non_contig_memory_format = non_contig.suggest_memory_format();
                auto non_contig_storage_offset = non_contig.storage_offset();
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
