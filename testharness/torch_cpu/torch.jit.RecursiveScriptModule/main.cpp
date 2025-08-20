#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <sstream>        // For stringstream

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a simple tensor to use with the script module
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple script module
        std::string script_code = R"(
def forward(x):
    if x.dim() == 0:
        return x + 1
    elif x.dim() == 1:
        return x.sum()
    else:
        return x.mean()
        )";
        
        try {
            // Create a TorchScript module from Python code
            auto compilation_unit = torch::jit::compile(script_code);
            auto module = compilation_unit->find_function("forward");
            
            if (module) {
                // Try to run the module with our input tensor
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                
                // Execute the module
                auto output = (*module)(inputs);
                
                // Try to access the output as a tensor
                if (output.isTensor()) {
                    torch::Tensor result = output.toTensor();
                }
            }
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors but don't discard the input
        }
        
        // Try with a different script that might trigger different behaviors
        if (offset + 1 < Size) {
            uint8_t variant = Data[offset++];
            
            std::string complex_script;
            
            // Use the variant byte to select different script implementations
            switch (variant % 3) {
                case 0:
                    complex_script = R"(
def forward(x):
    if x.size(0) > 0 and x.dim() > 1 and x.size(1) == 10:
        return x.sum()
    return x
                    )";
                    break;
                    
                case 1:
                    complex_script = R"(
def recursive_func(x, depth: int):
    if depth <= 0:
        return x
    if x.dim() > 0:
        return recursive_func(x.sum(dim=-1), depth-1)
    return x

def forward(x):
    return recursive_func(x, 3)
                    )";
                    break;
                    
                case 2:
                    complex_script = R"(
def forward(x):
    if x.dtype == torch.float32:
        return x * 2.0
    elif x.dtype == torch.int64:
        return x + 1
    elif x.dtype == torch.bool:
        return ~x
    else:
        return x
                    )";
                    break;
            }
            
            try {
                auto complex_compilation_unit = torch::jit::compile(complex_script);
                auto complex_module = complex_compilation_unit->find_function("forward");
                
                if (complex_module) {
                    std::vector<torch::jit::IValue> inputs;
                    inputs.push_back(input_tensor);
                    
                    auto output = (*complex_module)(inputs);
                    
                    if (output.isTensor()) {
                        torch::Tensor result = output.toTensor();
                    }
                }
            } catch (const c10::Error& e) {
                // Catch PyTorch-specific errors but don't discard the input
            }
        }
        
        // Test serialization and deserialization of script functions
        if (offset + 1 < Size) {
            try {
                std::string simple_script = R"(
def forward(x):
    return x + 1
                )";
                
                auto compilation_unit = torch::jit::compile(simple_script);
                auto module = compilation_unit->find_function("forward");
                
                if (module) {
                    // Try to run the module
                    std::vector<torch::jit::IValue> inputs;
                    inputs.push_back(input_tensor);
                    
                    auto output = (*module)(inputs);
                    
                    if (output.isTensor()) {
                        torch::Tensor result = output.toTensor();
                    }
                }
            } catch (const c10::Error& e) {
                // Catch PyTorch-specific errors but don't discard the input
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