#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor and define a script function
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor from fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Define a simple script function
        std::string script_code;
        
        // Use remaining bytes to determine which function to test
        if (offset < Size) {
            uint8_t func_selector = Data[offset++] % 5;
            
            switch (func_selector) {
                case 0:
                    script_code = "def forward(x):\n  return x.sin()";
                    break;
                case 1:
                    script_code = "def forward(x):\n  return x.abs() + 1";
                    break;
                case 2:
                    script_code = "def forward(x):\n  return x * x";
                    break;
                case 3:
                    script_code = "def forward(x):\n  return x.sum(dim=0) if x.dim() > 0 else x";
                    break;
                case 4:
                default:
                    script_code = "def forward(x):\n  return x.exp() if x.numel() > 0 else x";
                    break;
            }
        } else {
            script_code = "def forward(x):\n  return x + 1";
        }
        
        // Compile the script function
        std::shared_ptr<torch::jit::CompilationUnit> cu;
        try {
            cu = torch::jit::compile(script_code);
        } catch (const c10::Error& e) {
            // If compilation fails, try a simpler function
            script_code = "def forward(x):\n  return x";
            cu = torch::jit::compile(script_code);
        }
        
        // Get the forward function from the compilation unit
        torch::jit::Function& forward_func = cu->get_function("forward");
        
        // Create inputs vector for the function
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        // Call the function
        torch::jit::IValue output = forward_func(inputs);
        
        // Try to extract the tensor from the output
        if (output.isTensor()) {
            torch::Tensor result = output.toTensor();
            
            // Optional: perform some operation on the result to ensure it's used
            if (result.defined() && result.numel() > 0) {
                auto sum = result.sum().item<float>();
                if (std::isnan(sum) || std::isinf(sum)) {
                    // This is not an error, just an observation
                }
            }
        }
    }
    catch (const c10::Error &e)
    {
        // PyTorch specific errors
        return 0; // Keep the input for further analysis
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
