#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.special.expm1 operation
        torch::Tensor result = torch::special::expm1(input);
        
        // Try to access the result to ensure computation is performed
        if (result.defined()) {
            auto item = result.item();
        }
        
        // Try with out parameter variant
        if (offset + 1 < Size) {
            torch::Tensor output = torch::empty_like(input);
            torch::special::expm1_out(output, input);
            
            // Verify output has same shape as input
            if (output.sizes() != input.sizes()) {
                throw std::runtime_error("Output tensor shape mismatch");
            }
        }
        
        // Try with different dtypes if we have more data
        if (offset + 2 < Size) {
            // Try with float32
            torch::Tensor float_input = input.to(torch::kFloat32);
            torch::Tensor float_result = torch::special::expm1(float_input);
            
            // Try with float64
            torch::Tensor double_input = input.to(torch::kFloat64);
            torch::Tensor double_result = torch::special::expm1(double_input);
        }
        
        // Try with edge cases if we have more data
        if (offset + 3 < Size) {
            // Create tensors with special values
            torch::Tensor special_values = torch::tensor({-INFINITY, -1000.0, -1.0, -0.0, 0.0, 1.0, 1000.0, INFINITY, NAN});
            torch::Tensor special_result = torch::special::expm1(special_values);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}