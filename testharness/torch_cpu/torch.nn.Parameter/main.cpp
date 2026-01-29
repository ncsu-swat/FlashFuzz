#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

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
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor is floating point for gradient operations
        if (!tensor.is_floating_point()) {
            tensor = tensor.to(torch::kFloat32);
        }
        
        // Create a Parameter from the tensor
        // Test with different requires_grad values
        bool requires_grad = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
        
        // In C++ frontend, a "Parameter" is just a tensor with requires_grad set
        // We clone and set requires_grad to simulate Parameter behavior
        torch::Tensor parameter = tensor.clone().set_requires_grad(requires_grad);
        
        // Test basic Parameter properties
        auto param_data = parameter.data();
        
        // Test Parameter operations
        if (requires_grad && parameter.requires_grad()) {
            try {
                // Create a simple operation to generate gradients
                torch::Tensor output = parameter.mean();
                
                // Backpropagate
                output.backward();
                
                // Access the gradient
                auto grad_after = parameter.grad();
                
                // Test if gradient was properly computed
                if (grad_after.defined()) {
                    // Perform some operation with the gradient
                    auto grad_sum = grad_after.sum();
                    (void)grad_sum;
                }
            } catch (const std::exception &) {
                // Silently catch backward errors (e.g., complex tensors)
            }
        }
        
        // Test cloning
        auto cloned_param = parameter.clone();
        (void)cloned_param;
        
        // Test detach
        auto detached = parameter.detach();
        (void)detached;
        
        // Test to method (changing device/dtype)
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            try {
                auto dtype = fuzzer_utils::parseDataType(dtype_selector);
                // Only convert to floating point types to maintain requires_grad compatibility
                if (dtype == torch::kFloat32 || dtype == torch::kFloat64 || dtype == torch::kFloat16) {
                    auto converted = parameter.to(dtype);
                    (void)converted;
                }
            } catch (const std::exception &) {
                // Silently catch dtype conversion errors
            }
        }
        
        // Test Parameter with empty tensor
        if (offset < Size && Data[offset++] % 5 == 0) {
            try {
                std::vector<int64_t> empty_shape = {0};
                auto empty_tensor = torch::empty(empty_shape, torch::kFloat32);
                torch::Tensor empty_param = empty_tensor.set_requires_grad(requires_grad);
                (void)empty_param;
            } catch (const std::exception &) {
                // Silently catch errors with empty tensors
            }
        }
        
        // Test Parameter with scalar tensor
        if (offset < Size && Data[offset++] % 5 == 0) {
            auto scalar_tensor = torch::tensor(3.14f);
            torch::Tensor scalar_param = scalar_tensor.set_requires_grad(requires_grad);
            (void)scalar_param;
        }
        
        // Test Parameter with multi-dimensional tensor
        if (offset < Size && Data[offset++] % 4 == 0) {
            try {
                int64_t dim1 = (offset < Size) ? (Data[offset++] % 8 + 1) : 4;
                int64_t dim2 = (offset < Size) ? (Data[offset++] % 8 + 1) : 4;
                auto multi_tensor = torch::randn({dim1, dim2});
                torch::Tensor multi_param = multi_tensor.set_requires_grad(true);
                
                // Test gradient accumulation
                auto out1 = multi_param.sum();
                out1.backward();
                auto out2 = (multi_param * 2).sum();
                out2.backward();
                
                (void)multi_param;
            } catch (const std::exception &) {
                // Silently catch errors
            }
        }
        
        // Test zero_grad equivalent (set grad to undefined or zeros)
        if (parameter.grad().defined()) {
            parameter.mutable_grad().zero_();
        }
        
        // Test Parameter in a simple module context
        if (offset < Size && Data[offset++] % 3 == 0) {
            try {
                // Create a simple linear module to test Parameter integration
                int64_t in_features = (offset < Size) ? (Data[offset++] % 16 + 1) : 8;
                int64_t out_features = (offset < Size) ? (Data[offset++] % 16 + 1) : 4;
                
                torch::nn::Linear linear(in_features, out_features);
                
                // Access parameters
                for (auto& param : linear->parameters()) {
                    auto p_data = param.data();
                    (void)p_data;
                }
            } catch (const std::exception &) {
                // Silently catch module errors
            }
        }
        
        // Test registering a parameter in a custom module
        if (offset < Size && Data[offset++] % 4 == 0) {
            try {
                struct TestModule : torch::nn::Module {
                    TestModule(torch::Tensor t) {
                        // register_parameter is the C++ way to add a parameter
                        param = register_parameter("weight", t);
                    }
                    torch::Tensor param;
                };
                
                auto test_tensor = torch::randn({4, 4});
                TestModule module(test_tensor);
                
                // Verify parameter was registered
                auto params = module.parameters();
                for (auto& p : params) {
                    (void)p.data();
                }
            } catch (const std::exception &) {
                // Silently catch module errors
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