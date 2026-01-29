#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

// Simple test module to demonstrate parameter registration
struct TestModule : torch::nn::Module {
    TestModule(torch::Tensor init_tensor) {
        // Register a parameter with the module
        param = register_parameter("weight", init_tensor.clone().set_requires_grad(true));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        return x * param;
    }
    
    torch::Tensor param;
};

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
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor is floating point for gradient operations
        if (!tensor.is_floating_point()) {
            tensor = tensor.to(torch::kFloat32);
        }
        
        // Test 1: Basic requires_grad functionality (Parameter-like behavior)
        bool requires_grad = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
        torch::Tensor param_tensor = tensor.clone().set_requires_grad(requires_grad);
        
        // Verify requires_grad is set
        bool has_grad = param_tensor.requires_grad();
        (void)has_grad;
        
        // Test 2: Create a module with registered parameter
        try {
            auto module = std::make_shared<TestModule>(tensor);
            
            // Access named parameters
            auto params = module->named_parameters();
            for (auto& p : params) {
                auto name = p.key();
                auto value = p.value();
                bool p_requires_grad = value.requires_grad();
                (void)name;
                (void)p_requires_grad;
            }
            
            // Test forward pass with gradient tracking
            torch::Tensor input = torch::ones({tensor.size(0)});
            torch::Tensor output = module->forward(input);
            
            // Test backward pass if requires_grad
            if (module->param.requires_grad()) {
                torch::Tensor loss = output.sum();
                loss.backward();
                
                // Access gradient
                if (module->param.grad().defined()) {
                    auto grad_data = module->param.grad();
                    (void)grad_data;
                }
            }
            
            // Test parameter cloning
            auto cloned = module->param.clone();
            (void)cloned;
            
            // Test parameter detach
            auto detached = module->param.detach();
            (void)detached;
            
            // Test zero_grad
            module->zero_grad();
            
        } catch (const std::exception&) {
            // Module creation might fail with certain tensor shapes
        }
        
        // Test 3: Manual parameter operations
        if (requires_grad && param_tensor.numel() > 0) {
            try {
                torch::Tensor result = param_tensor.mean();
                result.backward();
                
                if (param_tensor.grad().defined()) {
                    // Test grad() accessor
                    auto g = param_tensor.grad();
                    (void)g;
                    
                    // Test is_leaf
                    bool is_leaf = param_tensor.is_leaf();
                    (void)is_leaf;
                }
            } catch (const std::exception&) {
                // Backward might fail for certain configurations
            }
        }
        
        // Test 4: Parameter data access and modification
        {
            auto data = param_tensor.data();
            (void)data;
            
            // Test requires_grad (setter)
            auto another = tensor.clone();
            another.set_requires_grad(true);
            another.set_requires_grad(false);
        }
        
        // Test 5: Multiple parameters interaction
        if (offset + 2 < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            if (!tensor2.is_floating_point()) {
                tensor2 = tensor2.to(torch::kFloat32);
            }
            
            torch::Tensor param2 = tensor2.clone().set_requires_grad(true);
            
            try {
                // Test operations between parameters
                if (param_tensor.sizes() == param2.sizes()) {
                    auto sum = param_tensor + param2;
                    auto prod = param_tensor * param2;
                    (void)prod;
                    
                    if (sum.requires_grad()) {
                        sum.sum().backward();
                    }
                }
            } catch (const std::exception&) {
                // Shape mismatch or other expected errors
            }
        }
        
        // Test 6: Parameter properties
        bool is_contiguous = param_tensor.is_contiguous();
        auto dtype = param_tensor.dtype();
        auto device = param_tensor.device();
        auto sizes = param_tensor.sizes();
        auto strides = param_tensor.strides();
        int64_t numel = param_tensor.numel();
        int64_t dim = param_tensor.dim();
        (void)is_contiguous;
        (void)dtype;
        (void)device;
        (void)sizes;
        (void)strides;
        (void)numel;
        (void)dim;
        
        // Test 7: retain_grad for non-leaf tensors
        if (requires_grad && param_tensor.numel() > 0) {
            try {
                torch::Tensor non_leaf = param_tensor * 2;
                non_leaf.retain_grad();
                non_leaf.sum().backward();
                
                if (non_leaf.grad().defined()) {
                    auto retained = non_leaf.grad();
                    (void)retained;
                }
            } catch (const std::exception&) {
                // May fail in certain conditions
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