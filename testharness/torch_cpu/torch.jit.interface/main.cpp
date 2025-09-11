#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h>
#include <torch/custom_class.h>

// Define a simple interface
struct MyModuleInterface : torch::CustomClassHolder {
    virtual torch::Tensor forward(torch::Tensor x) = 0;
};

// Implement the interface
struct MyModule : MyModuleInterface {
    torch::Tensor forward(torch::Tensor x) override {
        return x + 1;
    }
};

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Register the interface and implementation
        c10::intrusive_ptr<MyModuleInterface> impl = c10::make_intrusive<MyModule>();
        
        // Test interface registration
        torch::registerCustomClass<MyModuleInterface>("MyModuleInterface");
        torch::registerCustomClass<MyModule>("MyModule");
        
        // Create a script module that uses the interface
        std::string script_code = R"(
            class TestModule(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.interface = None
                
                def forward(self, x):
                    if self.interface is not None:
                        return self.interface.forward(x)
                    return x
        )";
        
        try {
            auto cu = torch::jit::compile(script_code);
            auto class_type = cu->get_class("TestModule");
            auto test_module = torch::jit::Module(class_type->create_instance());
            
            // Set the interface implementation
            test_module.setattr("interface", c10::IValue(impl));
            
            // Call the module with our tensor
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            
            auto output = test_module.forward(inputs);
            
            // Try to get the tensor result
            if (output.isTensor()) {
                torch::Tensor result = output.toTensor();
            }
        } catch (const c10::Error& e) {
            // Expected exceptions from torch::jit operations
        }
        
        // Try alternative interface usage patterns
        try {
            // Create a direct interface reference
            auto interface_type = torch::getCustomClass("__torch__.MyModuleInterface");
            if (interface_type) {
                // Just test that we can get the type
            }
        } catch (const c10::Error& e) {
            // Expected exceptions
        }
        
        // Try with different interface method signatures
        try {
            struct AnotherInterface : torch::CustomClassHolder {
                virtual torch::Tensor process(torch::Tensor x, int64_t value) = 0;
            };
            
            struct AnotherImpl : AnotherInterface {
                torch::Tensor process(torch::Tensor x, int64_t value) override {
                    return x * value;
                }
            };
            
            torch::registerCustomClass<AnotherInterface>("AnotherInterface");
            torch::registerCustomClass<AnotherImpl>("AnotherImpl");
            
            c10::intrusive_ptr<AnotherInterface> another_impl = c10::make_intrusive<AnotherImpl>();
            
            // Use the interface
            if (offset + 1 < Size) {
                int64_t value = static_cast<int64_t>(Data[offset]);
                another_impl->process(input_tensor, value);
            }
        } catch (const c10::Error& e) {
            // Expected exceptions
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
