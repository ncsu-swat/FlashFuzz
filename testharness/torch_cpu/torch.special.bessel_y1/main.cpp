#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // bessel_y1 requires floating point input
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat64);
        }
        
        // Apply torch.special.bessel_y1 operation
        torch::Tensor result = torch::special::bessel_y1(input);
        
        // Try to access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            auto first_element = result.flatten()[0].item<double>();
            (void)first_element; // Prevent unused variable warning
        }
        
        // Test the out= variant for additional coverage
        if (offset < Size) {
            torch::Tensor out_tensor = torch::empty_like(input);
            torch::special::bessel_y1_out(out_tensor, input);
            
            if (out_tensor.defined() && out_tensor.numel() > 0) {
                auto out_element = out_tensor.flatten()[0].item<double>();
                (void)out_element;
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