#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // torch.special.sinc works with floating point tensors
        // Convert to float if needed
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Apply torch.special.sinc operation
        // sinc(x) = sin(pi*x) / (pi*x), with sinc(0) = 1
        torch::Tensor result = torch::special::sinc(input);
        
        // Verify the result is computed (access first element if exists)
        if (result.defined() && result.numel() > 0) {
            // Use sum() instead of item() to handle multi-element tensors
            volatile float check = result.sum().item<float>();
            (void)check;
        }
        
        // Test with out parameter variant if we have enough data
        if (offset < Size) {
            torch::Tensor out = torch::empty_like(input);
            torch::special::sinc_out(out, input);
            
            if (out.defined() && out.numel() > 0) {
                volatile float check = out.sum().item<float>();
                (void)check;
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