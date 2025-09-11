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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.special.i1 operation
        torch::Tensor result = torch::special::i1(input);
        
        // Try to access the result to ensure computation is performed
        if (result.defined()) {
            auto item = result.item();
        }
        
        // Try with different input configurations if there's more data
        if (offset + 2 < Size) {
            torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            torch::Tensor result2 = torch::special::i1(input2);
            
            // Try to access the result to ensure computation is performed
            if (result2.defined()) {
                auto item2 = result2.item();
            }
        }
        
        // Test with edge cases if possible
        if (input.numel() > 0) {
            // Test with extreme values
            torch::Tensor extreme_values = torch::tensor({
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::min(),
                std::numeric_limits<float>::lowest(),
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::quiet_NaN()
            });
            
            torch::Tensor extreme_result = torch::special::i1(extreme_values);
            
            // Test with zero
            torch::Tensor zero_tensor = torch::zeros_like(input);
            torch::Tensor zero_result = torch::special::i1(zero_tensor);
            
            // Test with very small values
            torch::Tensor small_values = torch::full_like(input, 1e-10);
            torch::Tensor small_result = torch::special::i1(small_values);
            
            // Test with very large values
            torch::Tensor large_values = torch::full_like(input, 1e10);
            torch::Tensor large_result = torch::special::i1(large_values);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
