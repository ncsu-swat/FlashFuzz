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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.special.bessel_y0 operation
        torch::Tensor result = torch::special::bessel_y0(input);
        
        // Optionally, perform additional operations with the result
        // to ensure the operation is fully executed
        if (!result.is_floating_point()) {
            // This should not happen as bessel_y0 returns floating point,
            // but we check anyway to ensure the operation completed
            return 0;
        }
        
        // Check if result contains any NaN or Inf values
        // This is useful to detect numerical instabilities
        auto is_nan = torch::isnan(result);
        auto is_inf = torch::isinf(result);
        
        // Sum the result to force evaluation of all elements
        auto sum = result.sum().item<double>();
        
        // Use the result to prevent the compiler from optimizing away the operation
        if (std::isnan(sum) || std::isinf(sum)) {
            // We don't actually need to do anything special here,
            // just making sure the computation is not optimized away
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
