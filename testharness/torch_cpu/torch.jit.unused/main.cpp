#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h>

// Define a simple script module with @torch.jit.unused annotation
class TestModuleWithUnused : public torch::CustomClassHolder {
public:
    TestModuleWithUnused() {}

    torch::Tensor forward(torch::Tensor x) {
        if (x.size(0) > 0) {
            return x;
        } else {
            return unused_method(x);
        }
    }

    torch::Tensor unused_method(torch::Tensor x) {
        throw std::runtime_error("This method should not be called in TorchScript");
    }
};

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor from the fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Try to script the module (this should work despite the unused method)
        torch::jit::Module scripted_module;
        try {
            scripted_module = torch::jit::compile(R"(
                def forward(x):
                    if x.size(0) > 0:
                        return x
                    else:
                        return unused_method(x)
                
                @torch.jit.unused
                def unused_method(x):
                    # This method is marked as unused and should be ignored during scripting
                    return x
            )");
            
            // Try to run the forward method with the input tensor
            torch::Tensor output = scripted_module.forward({input_tensor}).toTensor();
        } catch (const c10::Error& e) {
            // Expected behavior for empty tensors - the unused method would be called
            // but it's not available in TorchScript
        }
        
        // Test direct usage of torch::jit::unused
        try {
            // Create a custom class with unused methods
            auto custom_class = torch::class_<TestModuleWithUnused>("TestModuleWithUnused")
                .def("forward", &TestModuleWithUnused::forward)
                .def("unused_method", &TestModuleWithUnused::unused_method);
                
            // Try to use the module with the input tensor
            auto instance = c10::make_intrusive<TestModuleWithUnused>();
            torch::Tensor output = instance->forward(input_tensor);
        } catch (const c10::Error& e) {
            // Expected for certain inputs
        }
        
        // Test with a simple TorchScript module that uses @torch.jit.unused
        try {
            auto module = torch::jit::compile(R"(
                def test_unused():
                    return helper_used()
                
                def helper_used():
                    return torch.tensor([1.0])
                
                @torch.jit.unused
                def helper_unused():
                    return torch.tensor([2.0])
            )");
            
            auto result = module.run_method("test_unused");
        } catch (const c10::Error& e) {
            // Handle compilation or execution errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}