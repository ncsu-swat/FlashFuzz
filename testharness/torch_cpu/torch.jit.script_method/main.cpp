#include "fuzzer_utils.h"
#include <iostream>
#include <torch/script.h>

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
        
        if (Size < 3) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create another tensor if there's data left
        torch::Tensor input2;
        if (offset + 2 < Size) {
            input2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            input2 = torch::ones_like(input1);
        }
        
        // Select which method to test based on remaining data
        uint8_t method_selector = 0;
        if (offset < Size) {
            method_selector = Data[offset++] % 4;
        }
        
        // Test different scripted operations using torch::jit::compile
        try {
            switch (method_selector) {
                case 0: {
                    // Test simple forward function
                    auto cu = torch::jit::compile(R"(
                        def forward(x: Tensor) -> Tensor:
                            return x + 1
                    )");
                    auto output = cu->get_function("forward")({input1});
                    break;
                }
                case 1: {
                    // Test add function
                    auto cu = torch::jit::compile(R"(
                        def add(x: Tensor, y: Tensor) -> Tensor:
                            return x + y
                    )");
                    auto output = cu->get_function("add")({input1, input2});
                    break;
                }
                case 2: {
                    // Test multiply function
                    auto cu = torch::jit::compile(R"(
                        def multiply(x: Tensor, y: Tensor) -> Tensor:
                            return x * y
                    )");
                    auto output = cu->get_function("multiply")({input1, input2});
                    break;
                }
                case 3: {
                    // Test identity function
                    auto cu = torch::jit::compile(R"(
                        def identity(x: Tensor) -> Tensor:
                            return x
                    )");
                    auto output = cu->get_function("identity")({input1});
                    break;
                }
            }
            
            // Test with more complex script methods
            if (offset < Size && Data[offset] % 2 == 0) {
                auto cu = torch::jit::compile(R"(
                    def complex_method(x: Tensor, y: Tensor) -> Tensor:
                        z = x + y
                        if bool(z.sum() > 0):
                            return z * 2
                        else:
                            return z - 1
                )");
                auto output = cu->get_function("complex_method")({input1, input2});
            }
            
            // Test with control flow
            if (offset < Size && Data[offset] % 3 == 0) {
                auto cu = torch::jit::compile(R"(
                    def control_flow(x: Tensor) -> Tensor:
                        result = x
                        for i in range(3):
                            result = result + i
                        return result
                )");
                auto output = cu->get_function("control_flow")({input1});
            }
            
            // Test with nested function calls within compilation unit
            if (offset < Size && Data[offset] % 5 == 0) {
                auto cu = torch::jit::compile(R"(
                    def helper(x: Tensor) -> Tensor:
                        return x * 2
                        
                    def nested_calls(x: Tensor) -> Tensor:
                        return helper(x) + helper(x + 1)
                )");
                auto output = cu->get_function("nested_calls")({input1});
            }
            
            // Test Module with registered parameters
            if (offset < Size && Data[offset] % 7 == 0) {
                torch::jit::Module module("test_module");
                module.register_parameter("weight", torch::randn({3, 3}), false);
                module.define(R"(
                    def forward(self, x: Tensor) -> Tensor:
                        return torch.matmul(x, self.weight)
                )");
                
                // Create appropriately shaped input for matmul
                auto shaped_input = torch::randn({2, 3});
                auto output = module.forward({shaped_input});
            }
        }
        catch (const c10::Error& e) {
            // Expected PyTorch errors (shape mismatches, etc.) are fine
        }
        catch (const std::runtime_error& e) {
            // TorchScript compilation/runtime errors are expected
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}