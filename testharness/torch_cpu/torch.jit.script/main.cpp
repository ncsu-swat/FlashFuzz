#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h> // For torch::jit::script

// Simple function to be scripted
struct SimpleModule : torch::nn::Module {
    SimpleModule() {}
    
    torch::Tensor forward(torch::Tensor x) {
        return x + x;
    }
    
    torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b) {
        return a + b;
    }
    
    torch::Tensor multiply(torch::Tensor x, float factor) {
        return x * factor;
    }
};

// Function to create a simple script function
torch::jit::Module createScriptFunction(const uint8_t* data, size_t size, size_t& offset) {
    SimpleModule m;
    return torch::jit::trace(m, torch::randn({2, 2}));
}

// Function to create a simple script function from a string
torch::jit::Module createScriptFromString(const uint8_t* data, size_t size, size_t& offset) {
    std::string script_code;
    
    // Extract some bytes to determine script code variant
    if (offset < size) {
        uint8_t variant = data[offset++] % 3;
        
        switch (variant) {
            case 0:
                script_code = "def forward(x):\n  return x + x";
                break;
            case 1:
                script_code = "def forward(x, y):\n  return x * y";
                break;
            case 2:
                script_code = "def forward(x):\n  return x.sum(dim=0)";
                break;
        }
    } else {
        script_code = "def forward(x):\n  return x";
    }
    
    auto cu = torch::jit::compile(script_code);
    return torch::jit::Module();
}

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to do anything meaningful
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a second tensor if we have more data
        torch::Tensor second_tensor;
        if (offset + 2 < Size) {
            second_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            second_tensor = input_tensor.clone();
        }
        
        // Get a variant selector if we have data left
        uint8_t variant = 0;
        if (offset < Size) {
            variant = Data[offset++] % 5;
        }
        
        // Test different scripting scenarios based on variant
        try {
            // Create a simple module and script it
            SimpleModule module;
            
            switch (variant) {
                case 0: {
                    // Trace the entire module
                    auto scripted_module = torch::jit::trace(module, input_tensor);
                    
                    // Call the forward method
                    auto output = scripted_module.forward({input_tensor}).toTensor();
                    break;
                }
                
                case 1: {
                    // Trace the module and call forward method
                    auto scripted_module = torch::jit::trace(module, input_tensor);
                    auto output = scripted_module.forward({input_tensor}).toTensor();
                    break;
                }
                
                case 2: {
                    // Trace the module and call forward method
                    auto scripted_module = torch::jit::trace(module, input_tensor);
                    auto output = scripted_module.forward({input_tensor}).toTensor();
                    break;
                }
                
                case 3: {
                    // Create a script module from a string
                    auto script_module = createScriptFromString(Data, Size, offset);
                    
                    // Call the forward method with one tensor
                    if (offset < Size && Data[offset++] % 2 == 0) {
                        // Just use input tensor
                    } else {
                        // Just use input tensor
                    }
                    break;
                }
                
                case 4: {
                    // Create a simple traced function
                    auto traced_func = torch::jit::trace([](torch::Tensor x) -> torch::Tensor {
                        return x.sigmoid();
                    }, input_tensor);
                    auto output = traced_func.forward({input_tensor});
                    break;
                }
            }
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
