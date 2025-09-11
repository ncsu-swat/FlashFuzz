#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <vector>
#include <functional>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Make sure input requires grad
        input = input.detach().requires_grad_(true);
        
        // Create a simple function to compute gradients for
        auto func = [](const torch::Tensor& x) -> torch::Tensor {
            return x.pow(2).sum();
        };
        
        // Create a vector of inputs
        std::vector<torch::Tensor> inputs = {input};
        
        // Create outputs for grad
        std::vector<torch::Tensor> outputs;
        
        // Create grad_outputs (optional)
        std::vector<torch::Tensor> grad_outputs;
        
        // Try different combinations of parameters for torch::autograd::grad
        if (offset < Size) {
            uint8_t param_selector = Data[offset++];
            
            // Create output
            outputs.push_back(func(input));
            
            // Optional: create grad_outputs
            if (param_selector & 0x01) {
                if (offset < Size) {
                    torch::Tensor grad_output = fuzzer_utils::createTensor(Data, Size, offset);
                    
                    // Try to match the shape of outputs[0]
                    if (outputs[0].dim() > 0) {
                        try {
                            grad_output = grad_output.reshape_as(outputs[0]);
                        } catch (...) {
                            // If reshape fails, create a tensor with the right shape
                            grad_output = torch::ones_like(outputs[0]);
                        }
                    }
                    
                    grad_outputs.push_back(grad_output);
                }
            }
            
            // Set various options for grad
            bool create_graph = (param_selector & 0x02) != 0;
            bool retain_graph = (param_selector & 0x04) != 0;
            bool allow_unused = (param_selector & 0x08) != 0;
            
            // Call torch::autograd::grad with different parameter combinations
            try {
                std::vector<torch::Tensor> gradients;
                
                if (grad_outputs.empty()) {
                    gradients = torch::autograd::grad(
                        outputs,
                        inputs,
                        {},
                        create_graph,
                        retain_graph,
                        allow_unused
                    );
                } else {
                    gradients = torch::autograd::grad(
                        outputs,
                        inputs,
                        grad_outputs,
                        create_graph,
                        retain_graph,
                        allow_unused
                    );
                }
                
                // Do something with the gradients to ensure they're used
                if (!gradients.empty() && gradients[0].defined()) {
                    auto sum = gradients[0].sum();
                    
                    // If create_graph is true, we can compute higher-order gradients
                    if (create_graph && sum.requires_grad()) {
                        auto grad_grad = torch::autograd::grad(
                            {sum},
                            {input},
                            {},
                            false,
                            retain_graph,
                            allow_unused
                        );
                    }
                }
            } catch (const c10::Error& e) {
                // PyTorch-specific errors are expected in some cases
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
