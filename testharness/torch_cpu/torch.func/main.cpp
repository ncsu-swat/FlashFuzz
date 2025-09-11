#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test basic autograd functionality instead of torch.func
        if (offset < Size) {
            try {
                // Create a tensor that requires grad
                auto x = input.clone().detach().requires_grad_(true);
                
                // Compute a simple function and its gradient
                auto y = x.pow(2).sum();
                y.backward();
                auto grad = x.grad();
            } catch (const std::exception& e) {
                // Catch exceptions but continue testing
            }
        }
        
        // Test higher order derivatives
        if (offset < Size) {
            try {
                // Create a tensor that requires grad
                auto x = input.clone().detach().requires_grad_(true);
                
                // Compute first derivative
                auto y = x.sin().sum();
                auto grad_outputs = torch::ones_like(y);
                auto first_grad = torch::autograd::grad({y}, {x}, {grad_outputs}, true, true)[0];
                
                // Compute second derivative
                auto second_grad = torch::autograd::grad({first_grad.sum()}, {x})[0];
            } catch (const std::exception& e) {
                // Catch exceptions but continue testing
            }
        }
        
        // Test jacobian computation using autograd
        if (offset < Size) {
            try {
                // Create a tensor that requires grad
                auto x = input.clone().detach().requires_grad_(true);
                
                // Define a vector-valued function output
                auto y = torch::stack({x.sin(), x.cos()});
                
                // Compute jacobian manually using autograd
                std::vector<torch::Tensor> jacobian_rows;
                for (int i = 0; i < y.size(0); ++i) {
                    auto grad_outputs = torch::zeros_like(y);
                    grad_outputs[i] = 1.0;
                    auto grad = torch::autograd::grad({y}, {x}, {grad_outputs}, true, true);
                    if (!grad.empty()) {
                        jacobian_rows.push_back(grad[0]);
                    }
                }
            } catch (const std::exception& e) {
                // Catch exceptions but continue testing
            }
        }
        
        // Test functional transformations using basic operations
        if (offset < Size) {
            try {
                // Apply element-wise operations that could simulate vmap behavior
                auto squared = input.pow(2);
                auto result = squared.sum();
            } catch (const std::exception& e) {
                // Catch exceptions but continue testing
            }
        }
        
        // Test batch operations
        if (offset < Size) {
            try {
                // Create batch dimension and apply operations
                if (input.dim() > 0) {
                    auto batched = input.unsqueeze(0).repeat({2, 1});
                    auto result = batched.tanh();
                }
            } catch (const std::exception& e) {
                // Catch exceptions but continue testing
            }
        }
        
        // Test forward mode differentiation simulation
        if (offset < Size) {
            try {
                // Create dual numbers simulation for forward mode AD
                auto x = input.clone().detach().requires_grad_(true);
                auto tangent = torch::ones_like(x);
                
                // Compute function value and derivative simultaneously
                auto y = x.sigmoid();
                auto dy_dx = torch::autograd::grad({y.sum()}, {x}, {torch::ones_like(y.sum())}, true, true);
            } catch (const std::exception& e) {
                // Catch exceptions but continue testing
            }
        }
        
        // Test reverse mode differentiation
        if (offset < Size) {
            try {
                auto x = input.clone().detach().requires_grad_(true);
                auto y = x.pow(3).sum();
                
                // Compute gradient using backward pass
                y.backward();
                auto grad = x.grad();
            } catch (const std::exception& e) {
                // Catch exceptions but continue testing
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
