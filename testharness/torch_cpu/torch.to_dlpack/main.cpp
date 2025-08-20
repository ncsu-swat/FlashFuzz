#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>
#include <ATen/DLConvertor.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert tensor to DLPack format
        DLManagedTensor* dlpack_tensor = at::toDLPack(tensor);
        
        // Convert back from DLPack to PyTorch tensor to verify round-trip
        torch::Tensor tensor_from_dlpack = at::fromDLPack(dlpack_tensor);
        
        // Verify that the tensors are the same
        if (tensor.sizes() != tensor_from_dlpack.sizes() ||
            tensor.dtype() != tensor_from_dlpack.dtype()) {
            throw std::runtime_error("Tensor conversion mismatch");
        }
        
        // Try to access data to ensure it's valid
        if (tensor.numel() > 0 && tensor_from_dlpack.numel() > 0) {
            tensor_from_dlpack.item();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}