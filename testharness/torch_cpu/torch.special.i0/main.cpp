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
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for torch.special.i0
        // i0 works on floating point tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float if needed (i0 requires floating point)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Apply torch.special.i0 operation
        // i0(x) = modified Bessel function of the first kind, order 0
        torch::Tensor result = torch::special::i0(input);
        
        // Verify result is defined and has expected properties
        if (result.defined()) {
            auto sizes = result.sizes();
            auto dtype = result.dtype();
            
            // Force evaluation of the tensor by computing sum
            if (result.numel() > 0) {
                volatile float sum = result.sum().item<float>();
                (void)sum;
            }
        }
        
        // Try with out variant if we have enough data left
        if (offset + 2 < Size) {
            // Create output tensor with same shape and dtype as input
            torch::Tensor output = torch::empty_like(input);
            
            // Apply torch.special.i0 with out parameter
            torch::special::i0_out(output, input);
            
            // Force evaluation
            if (output.numel() > 0) {
                volatile float sum = output.sum().item<float>();
                (void)sum;
            }
        }
        
        // Test with different tensor configurations to improve coverage
        if (offset + 4 < Size) {
            // Test with contiguous tensor
            torch::Tensor contiguous_input = input.contiguous();
            torch::Tensor result2 = torch::special::i0(contiguous_input);
            (void)result2;
            
            // Test with non-contiguous tensor (transposed if 2D+)
            if (input.dim() >= 2) {
                torch::Tensor transposed = input.transpose(0, 1);
                torch::Tensor result3 = torch::special::i0(transposed);
                (void)result3;
            }
        }
        
        // Test with different dtypes for better coverage
        if (offset + 6 < Size) {
            try {
                // Test with double precision
                torch::Tensor double_input = input.to(torch::kFloat64);
                torch::Tensor result_double = torch::special::i0(double_input);
                (void)result_double;
            } catch (...) {
                // Silently handle dtype conversion failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}