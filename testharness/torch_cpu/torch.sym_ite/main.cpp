#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 3 tensors for where: condition, x, y
        if (Size < 6) // Minimum bytes needed for basic tensor creation
            return 0;
        
        // Create condition tensor (should be boolean)
        torch::Tensor condition = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create x tensor (first result if condition is true)
        if (offset >= Size)
            return 0;
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create y tensor (second result if condition is false)
        if (offset >= Size)
            return 0;
        torch::Tensor y = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert condition to boolean if it's not already
        if (condition.dtype() != torch::kBool) {
            condition = condition.to(torch::kBool);
        }
        
        // Try to make x and y have compatible dtypes if possible
        if (x.dtype() != y.dtype()) {
            // Attempt to promote to a common dtype
            try {
                auto common_dtype = torch::promote_types(x.scalar_type(), y.scalar_type());
                x = x.to(common_dtype);
                y = y.to(common_dtype);
            } catch (...) {
                // If promotion fails, just continue with original tensors
            }
        }
        
        // Apply where operation (equivalent to sym_ite)
        torch::Tensor result = torch::where(condition, x, y);
        
        // Optionally test some properties of the result
        auto result_shape = result.sizes();
        auto result_dtype = result.dtype();
        
        // Access some elements to ensure computation is performed
        if (result.numel() > 0) {
            result.item();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}