#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Make the tensor require gradients
        input_tensor = input_tensor.detach().requires_grad_(true);
        
        // Test various autograd operations
        if (offset + 1 < Size) {
            uint8_t op_selector = Data[offset++];
            
            // Perform different autograd operations based on the selector
            switch (op_selector % 5) {
                case 0: {
                    // Test backward on a simple operation
                    auto output = input_tensor.sum();
                    output.backward();
                    auto grad = input_tensor.grad();
                    break;
                }
                case 1: {
                    // Test backward with gradient accumulation
                    auto output1 = input_tensor.pow(2).sum();
                    output1.backward(torch::ones_like(output1), true);
                    
                    auto output2 = input_tensor.exp().sum();
                    output2.backward(torch::ones_like(output2), true);
                    
                    auto grad = input_tensor.grad();
                    break;
                }
                case 2: {
                    // Test with_no_grad
                    torch::NoGradGuard no_grad;
                    auto output = input_tensor + input_tensor;
                    
                    // This should have no effect since we're in no_grad mode
                    if (output.requires_grad()) {
                        auto sum_output = output.sum();
                        sum_output.backward();
                    }
                    break;
                }
                case 3: {
                    // Test set_grad_enabled
                    bool prev = torch::autograd::GradMode::is_enabled();
                    torch::autograd::GradMode::set_enabled(false);
                    
                    auto output = input_tensor * 2;
                    
                    // Restore previous state
                    torch::autograd::GradMode::set_enabled(prev);
                    break;
                }
                case 4: {
                    // Test grad() function
                    if (input_tensor.dim() > 0 && input_tensor.numel() > 0) {
                        std::vector<torch::Tensor> inputs = {input_tensor};
                        std::vector<torch::Tensor> outputs;
                        
                        // Create a simple function to compute gradients for
                        auto output = input_tensor.sin();
                        outputs.push_back(output.sum());
                        
                        // Compute gradients
                        auto gradients = torch::autograd::grad(outputs, inputs);
                    }
                    break;
                }
            }
        }
        
        // Test creating a variable that doesn't require gradients
        if (offset < Size) {
            torch::Tensor no_grad_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Test operations mixing tensors with and without gradients
            if (input_tensor.numel() > 0 && no_grad_tensor.numel() > 0) {
                try {
                    // Try to make shapes compatible if possible
                    if (input_tensor.dim() > 0 && no_grad_tensor.dim() > 0) {
                        no_grad_tensor = no_grad_tensor.reshape({-1}).expand_as(input_tensor);
                    }
                    
                    auto result = input_tensor + no_grad_tensor;
                    auto output = result.sum();
                    output.backward();
                } catch (const std::exception&) {
                    // Shape mismatch or other error, continue
                }
            }
        }
        
        // Test autograd::Function API
        if (offset < Size) {
            class CustomFunction : public torch::autograd::Function<CustomFunction> {
            public:
                static torch::Tensor forward(
                    torch::autograd::AutogradContext *ctx,
                    const torch::Tensor& input) {
                    
                    ctx->save_for_backward({input});
                    return input.clone();
                }
                
                static torch::autograd::tensor_list backward(
                    torch::autograd::AutogradContext *ctx,
                    torch::autograd::tensor_list grad_outputs) {
                    
                    auto saved = ctx->get_saved_variables();
                    auto input = saved[0];
                    
                    return {grad_outputs[0].clone()};
                }
            };
            
            auto result = CustomFunction::apply(input_tensor);
            auto output = result.sum();
            output.backward();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}