#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

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
        
        // Create input tensor
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test various tensor utility functionality
        
        // 1. Test tensor device operations
        try {
            auto device = tensor.device();
            auto device_type = device.type();
            auto device_index = device.index();
        } catch (...) {
            // Continue with other tests even if this one fails
        }
        
        // 2. Test tensor options
        try {
            auto options = tensor.options();
            auto dtype = options.dtype();
            auto device = options.device();
        } catch (...) {
            // Continue with other tests
        }
        
        // 3. Test tensor properties
        try {
            auto sizes = tensor.sizes();
            auto strides = tensor.strides();
            auto numel = tensor.numel();
        } catch (...) {
            // Continue with other tests
        }
        
        // 4. Test tensor data access
        try {
            if (tensor.is_contiguous() && tensor.dtype() == torch::kFloat) {
                auto data_ptr = tensor.data_ptr<float>();
            }
        } catch (...) {
            // Continue with other tests
        }
        
        // 5. Test tensor creation with options
        if (offset + 1 < Size) {
            try {
                auto new_tensor = torch::zeros({2, 2}, tensor.options());
            } catch (...) {
                // Continue with other tests
            }
        }
        
        // 6. Test tensor device queries
        try {
            bool is_cuda = tensor.is_cuda();
            bool is_cpu = tensor.device().is_cpu();
        } catch (...) {
            // Continue with other tests
        }
        
        // 7. Test tensor type queries
        try {
            bool is_floating_point = tensor.is_floating_point();
            bool is_complex = tensor.is_complex();
            bool is_signed = tensor.is_signed();
        } catch (...) {
            // Continue with other tests
        }
        
        // 8. Test tensor memory layout
        try {
            auto storage = tensor.storage();
            auto storage_offset = tensor.storage_offset();
        } catch (...) {
            // Continue with other tests
        }
        
        // 9. Test tensor cloning and copying
        try {
            auto cloned = tensor.clone();
            auto copied = tensor.to(tensor.device());
        } catch (...) {
            // Continue with other tests
        }
        
        // 10. Test tensor reshaping utilities
        try {
            auto flattened = tensor.flatten();
            auto reshaped = tensor.view({-1});
        } catch (...) {
            // Continue with other tests
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
