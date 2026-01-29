#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr, cout

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
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }

        size_t offset = 0;
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply the asinh_ operation in-place
        // This modifies the tensor directly
        tensor.asinh_();
        
        // Access result to ensure computation happens
        volatile float first_elem = tensor.numel() > 0 ? tensor.flatten()[0].item<float>() : 0.0f;
        (void)first_elem;

        // Test with a second tensor if we have more data
        if (offset + 2 < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            tensor2.asinh_();
            
            // Force computation
            volatile float elem2 = tensor2.numel() > 0 ? tensor2.flatten()[0].item<float>() : 0.0f;
            (void)elem2;
        }

        // Test with contiguous tensor
        try {
            torch::Tensor contiguous_tensor = tensor.contiguous();
            contiguous_tensor.asinh_();
        } catch (...) {
            // Silently ignore expected failures
        }

        // Test with different dtypes based on fuzzer data
        if (Size > 4) {
            uint8_t dtype_selector = Data[Size - 1] % 4;
            torch::Tensor typed_tensor;
            
            try {
                switch (dtype_selector) {
                    case 0:
                        typed_tensor = torch::randn({3, 3}, torch::kFloat32);
                        break;
                    case 1:
                        typed_tensor = torch::randn({3, 3}, torch::kFloat64);
                        break;
                    case 2:
                        typed_tensor = torch::randn({2, 2, 2}, torch::kFloat32);
                        break;
                    case 3:
                        typed_tensor = torch::randn({4}, torch::kFloat64);
                        break;
                }
                typed_tensor.asinh_();
                
                // Force computation
                volatile double val = typed_tensor.flatten()[0].item<double>();
                (void)val;
            } catch (...) {
                // Silently ignore expected failures (e.g., dtype issues)
            }
        }

        // Test edge cases with special values
        if (Size > 2 && (Data[0] % 3 == 0)) {
            try {
                auto options = torch::TensorOptions().dtype(torch::kFloat32);
                
                torch::Tensor special_values = torch::tensor(
                    {std::numeric_limits<float>::infinity(), 
                     -std::numeric_limits<float>::infinity(),
                     std::numeric_limits<float>::quiet_NaN(),
                     0.0f, -0.0f, 1.0f, -1.0f,
                     100.0f, -100.0f, 1e-10f, -1e-10f}, options);
                
                special_values.asinh_();
                
                // Force computation
                volatile float special_val = special_values[0].item<float>();
                (void)special_val;
            } catch (...) {
                // Silently ignore expected failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}