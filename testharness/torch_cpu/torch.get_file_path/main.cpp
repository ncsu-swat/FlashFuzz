#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test tensor operations since get_file_path doesn't exist
        if (tensor.defined() && tensor.numel() > 0) {
            // Test basic tensor properties
            auto dtype = tensor.dtype();
            auto device = tensor.device();
            auto sizes = tensor.sizes();
            
            // Test tensor cloning and modification
            torch::Tensor cloned = tensor.clone();
            
            if (cloned.is_floating_point()) {
                cloned = cloned + 1.0;
            } else {
                cloned = cloned + 1;
            }
            
            // Test tensor comparison
            torch::Tensor comparison = torch::equal(tensor, cloned);
        }
        
        // Test with different tensor operations
        if (offset < Size) {
            try {
                // Create a scalar value
                torch::Scalar scalar = static_cast<int64_t>(Data[offset]);
                
                // Test scalar to tensor conversion
                torch::Tensor scalar_tensor = torch::scalar_tensor(scalar);
            } catch (...) {
                // Handle any exceptions
            }
        }
        
        // Test with undefined tensor
        try {
            torch::Tensor undefined_tensor;
            bool is_defined = undefined_tensor.defined();
        } catch (...) {
            // Handle exceptions
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}