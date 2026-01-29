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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // bessel_j1 requires floating point input
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Apply the bessel_j1 operation
        torch::Tensor result = torch::special::bessel_j1(input);
        
        // Access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            // Use sum() to access all elements without requiring single-element tensor
            volatile float check = result.sum().item<float>();
            (void)check;
        }
        
        // Try with different input configurations if we have more data
        if (Size - offset >= 2) {
            // Create another tensor with different properties
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (!input2.is_floating_point()) {
                input2 = input2.to(torch::kFloat64);
            }
            
            // Apply bessel_j1 to the second tensor
            torch::Tensor result2 = torch::special::bessel_j1(input2);
            
            // Access the result
            if (result2.defined() && result2.numel() > 0) {
                volatile double check = result2.sum().item<double>();
                (void)check;
            }
        }
        
        // Test with scalar input if we have more data
        if (Size - offset >= sizeof(double)) {
            double scalar_value;
            memcpy(&scalar_value, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Handle NaN/Inf from raw bytes - these are valid test cases
            torch::Tensor scalar_tensor = torch::tensor(scalar_value, torch::kFloat64);
            torch::Tensor scalar_result = torch::special::bessel_j1(scalar_tensor);
            
            if (scalar_result.defined() && scalar_result.numel() > 0) {
                volatile double check = scalar_result.item<double>();
                (void)check;
            }
        }
        
        // Test with extreme values
        if (Size - offset >= 1) {
            uint8_t selector = Data[offset++] % 5;
            torch::Tensor extreme_tensor;
            
            switch (selector) {
                case 0:
                    extreme_tensor = torch::tensor(std::numeric_limits<float>::infinity());
                    break;
                case 1:
                    extreme_tensor = torch::tensor(-std::numeric_limits<float>::infinity());
                    break;
                case 2:
                    extreme_tensor = torch::tensor(std::numeric_limits<float>::quiet_NaN());
                    break;
                case 3:
                    extreme_tensor = torch::tensor(0.0f);
                    break;
                case 4:
                    // Test with very large value near overflow
                    extreme_tensor = torch::tensor(1e38f);
                    break;
            }
            
            torch::Tensor extreme_result = torch::special::bessel_j1(extreme_tensor);
            
            if (extreme_result.defined() && extreme_result.numel() > 0) {
                volatile float check = extreme_result.item<float>();
                (void)check;
            }
        }
        
        // Test with output tensor variant if we have enough data
        if (Size - offset >= 2) {
            torch::Tensor input3 = fuzzer_utils::createTensor(Data, Size, offset);
            if (!input3.is_floating_point()) {
                input3 = input3.to(torch::kFloat32);
            }
            
            // Pre-allocate output tensor
            torch::Tensor out = torch::empty_like(input3);
            torch::special::bessel_j1_out(out, input3);
            
            if (out.defined() && out.numel() > 0) {
                volatile float check = out.sum().item<float>();
                (void)check;
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