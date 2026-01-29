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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset < Size) {
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply torch.divide operation (divide is an alias for div)
            torch::Tensor result = torch::divide(input1, input2);
            
            // Test with different rounding modes
            if (offset < Size) {
                uint8_t mode_selector = Data[offset++];
                
                try {
                    // Test different rounding modes
                    switch (mode_selector % 3) {
                        case 0:
                            // No rounding mode (default - true division)
                            result = torch::divide(input1, input2);
                            break;
                        case 1:
                            // trunc mode - rounds towards zero
                            result = torch::divide(input1, input2, "trunc");
                            break;
                        case 2:
                            // floor mode - rounds towards negative infinity
                            result = torch::divide(input1, input2, "floor");
                            break;
                    }
                } catch (...) {
                    // Silently catch rounding mode errors
                }
                
                // Test out variant (torch::div_out is the correct function name)
                try {
                    torch::Tensor out = torch::empty_like(input1);
                    torch::div_out(out, input1, input2);
                } catch (...) {
                    // Silently catch shape mismatch errors
                }
            }
            
            // Test scalar variants if we have more data
            if (offset < Size) {
                double scalar_value = static_cast<double>(Data[offset++]);
                if (scalar_value == 0) {
                    scalar_value = 1.0; // Avoid trivial div by zero
                }
                
                // Test tensor / scalar
                torch::Tensor result_scalar = torch::divide(input1, scalar_value);
                
                // Test with rounding mode
                try {
                    torch::Tensor result_trunc = torch::divide(input1, scalar_value, "trunc");
                    torch::Tensor result_floor = torch::divide(input1, scalar_value, "floor");
                } catch (...) {
                    // Silently catch type errors
                }
                
                // Test scalar / tensor - create scalar tensor first
                torch::Tensor scalar_tensor = torch::scalar_tensor(scalar_value, input1.options());
                torch::Tensor result_scalar_first = torch::divide(scalar_tensor, input1);
            }
        } else {
            // If we don't have enough data for a second tensor, try scalar division
            double scalar_value = 1.0 + static_cast<double>(Size % 255); // Use size as scalar
            torch::Tensor result = torch::divide(input1, scalar_value);
        }
        
        // Try inplace division if we have more data
        if (offset + 1 < Size) {
            uint8_t inplace_flag = Data[offset++];
            if (inplace_flag % 2 == 0) {
                // Create a copy to avoid modifying the original tensor
                // Also convert to float for inplace division
                torch::Tensor input_copy = input1.clone().to(torch::kFloat);
                
                try {
                    // Check if we have enough data for a second tensor
                    if (offset + 2 < Size) {
                        torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
                        input_copy.divide_(input2);
                    } else {
                        // Try scalar inplace division
                        double scalar_value = 1.0 + static_cast<double>(Data[offset++]);
                        input_copy.divide_(scalar_value);
                    }
                } catch (...) {
                    // Silently catch inplace operation errors (e.g., type mismatch)
                }
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