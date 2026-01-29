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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor - selu requires floating point input
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a floating point tensor for selu
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Apply selu operation
        torch::Tensor output = torch::selu(input);
        
        // Consume result to prevent optimization
        volatile float sink = output.sum().item<float>();
        (void)sink;
        
        // Try inplace version if we have more data
        if (offset < Size) {
            bool use_inplace = Data[offset++] & 0x1;
            
            if (use_inplace) {
                // Create a copy for inplace operation (needs to be contiguous and float)
                torch::Tensor input_copy = input.clone();
                torch::selu_(input_copy);
                
                // Consume result
                volatile float sink2 = input_copy.sum().item<float>();
                (void)sink2;
            }
        }
        
        // Try with different tensor shapes and dtypes
        if (offset + 4 < Size) {
            uint8_t shape_selector = Data[offset++];
            uint8_t dtype_selector = Data[offset++];
            
            // Create tensors with different shapes
            torch::Tensor test_tensor;
            try {
                switch (shape_selector % 5) {
                    case 0:
                        // Scalar-like
                        test_tensor = torch::randn({1});
                        break;
                    case 1:
                        // 1D
                        test_tensor = torch::randn({16});
                        break;
                    case 2:
                        // 2D
                        test_tensor = torch::randn({4, 4});
                        break;
                    case 3:
                        // 3D
                        test_tensor = torch::randn({2, 4, 4});
                        break;
                    case 4:
                        // 4D (batch-like)
                        test_tensor = torch::randn({2, 3, 4, 4});
                        break;
                }
                
                // Try different floating point dtypes
                switch (dtype_selector % 3) {
                    case 0:
                        test_tensor = test_tensor.to(torch::kFloat32);
                        break;
                    case 1:
                        test_tensor = test_tensor.to(torch::kFloat64);
                        break;
                    case 2:
                        test_tensor = test_tensor.to(torch::kFloat16);
                        break;
                }
                
                torch::Tensor result = torch::selu(test_tensor);
                volatile float sink3 = result.sum().item<float>();
                (void)sink3;
            }
            catch (...) {
                // Silently ignore expected failures (e.g., unsupported dtype combinations)
            }
        }
        
        // Test edge cases with special values
        if (offset < Size && (Data[offset++] & 0x1)) {
            try {
                // Test with zeros
                torch::Tensor zeros = torch::zeros({4, 4});
                torch::Tensor zero_result = torch::selu(zeros);
                
                // Test with negative values
                torch::Tensor negatives = torch::full({4, 4}, -2.0);
                torch::Tensor neg_result = torch::selu(negatives);
                
                // Test with positive values
                torch::Tensor positives = torch::full({4, 4}, 2.0);
                torch::Tensor pos_result = torch::selu(positives);
                
                // Consume results
                volatile float sink4 = zero_result.sum().item<float>();
                volatile float sink5 = neg_result.sum().item<float>();
                volatile float sink6 = pos_result.sum().item<float>();
                (void)sink4;
                (void)sink5;
                (void)sink6;
            }
            catch (...) {
                // Silently ignore
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