#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Create Sigmoid module and apply it
        torch::nn::Sigmoid sigmoid_module;
        torch::Tensor output = sigmoid_module->forward(input);
        
        // Test functional version
        torch::Tensor output2 = torch::sigmoid(input);
        
        // Test in-place version on floating point tensors
        if (input.is_floating_point()) {
            torch::Tensor input_copy = input.clone();
            input_copy.sigmoid_();
        }
        
        // Test with different tensor types based on fuzzer data
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++] % 4;
            torch::Tensor typed_input;
            
            try {
                switch (dtype_selector) {
                    case 0:
                        typed_input = input.to(torch::kFloat32);
                        break;
                    case 1:
                        typed_input = input.to(torch::kFloat64);
                        break;
                    case 2:
                        typed_input = input.to(torch::kFloat16);
                        break;
                    default:
                        typed_input = input.to(torch::kBFloat16);
                        break;
                }
                torch::Tensor typed_output = torch::sigmoid(typed_input);
            } catch (...) {
                // Some dtype conversions may fail, that's expected
            }
        }
        
        // Test with gradients
        if (input.is_floating_point() && offset + 1 < Size) {
            bool requires_grad = Data[offset++] % 2 == 0;
            
            if (requires_grad) {
                try {
                    auto input_with_grad = input.clone().detach().set_requires_grad(true);
                    auto output_with_grad = sigmoid_module->forward(input_with_grad);
                    
                    if (output_with_grad.numel() > 0) {
                        auto sum = output_with_grad.sum();
                        sum.backward();
                        
                        // Access the gradient to ensure it was computed
                        auto grad = input_with_grad.grad();
                    }
                } catch (...) {
                    // Gradient computation may fail for some inputs
                }
            }
        }
        
        // Test with different tensor shapes
        if (offset + 4 < Size) {
            int dim1 = (Data[offset++] % 8) + 1;
            int dim2 = (Data[offset++] % 8) + 1;
            int dim3 = (Data[offset++] % 8) + 1;
            int dim4 = (Data[offset++] % 8) + 1;
            
            try {
                // 1D tensor
                auto t1 = torch::randn({dim1});
                torch::sigmoid(t1);
                
                // 2D tensor
                auto t2 = torch::randn({dim1, dim2});
                sigmoid_module->forward(t2);
                
                // 3D tensor
                auto t3 = torch::randn({dim1, dim2, dim3});
                torch::sigmoid(t3);
                
                // 4D tensor (batch of images)
                auto t4 = torch::randn({dim1, dim2, dim3, dim4});
                sigmoid_module->forward(t4);
            } catch (...) {
                // Shape-related failures are expected
            }
        }
        
        // Test with special values
        if (offset + 1 < Size) {
            uint8_t special_selector = Data[offset++] % 4;
            torch::Tensor special_input;
            
            try {
                switch (special_selector) {
                    case 0:
                        // Very large positive values
                        special_input = torch::full({4, 4}, 100.0);
                        break;
                    case 1:
                        // Very large negative values
                        special_input = torch::full({4, 4}, -100.0);
                        break;
                    case 2:
                        // Zeros
                        special_input = torch::zeros({4, 4});
                        break;
                    default:
                        // Mix of inf and nan
                        special_input = torch::tensor({{INFINITY, -INFINITY}, {NAN, 0.0}});
                        break;
                }
                torch::Tensor special_output = torch::sigmoid(special_input);
            } catch (...) {
                // Special value handling may throw
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}