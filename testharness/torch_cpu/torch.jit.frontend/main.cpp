#include "fuzzer_utils.h"
#include <iostream>
#include <torch/script.h>
#include <torch/csrc/jit/frontend/parser.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <ATen/core/stack.h>

static const char *kTorchJitFrontendKeyword = "torch.jit.frontend";

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    (void)kTorchJitFrontendKeyword;
    
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to use in our script
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract some bytes to determine script type and content
        if (offset + 2 >= Size) {
            return 0;
        }
        
        uint8_t script_type_byte = Data[offset++];
        uint8_t script_length_byte = Data[offset++];
        
        // Determine script length (limit to reasonable size)
        size_t script_length = script_length_byte % 100;
        if (offset + script_length > Size) {
            script_length = Size - offset;
        }
        
        // Extract script content - sanitize to valid ASCII for parsing
        std::string script_content;
        if (script_length > 0) {
            script_content.reserve(script_length);
            for (size_t i = 0; i < script_length; i++) {
                char c = static_cast<char>(Data[offset + i]);
                // Keep printable ASCII and common whitespace
                if ((c >= 32 && c <= 126) || c == '\n' || c == '\t') {
                    script_content.push_back(c);
                } else {
                    script_content.push_back(' ');
                }
            }
            offset += script_length;
        }
        
        // Ensure we have some content
        if (script_content.empty() || script_content.find_first_not_of(" \t\n") == std::string::npos) {
            script_content = "def forward(x):\n  return x + 1";
        }
        
        // Try different frontend operations based on script_type_byte
        switch (script_type_byte % 4) {
            case 0: {
                // Parse source code using the parser
                try {
                    auto src = std::make_shared<torch::jit::Source>(script_content);
                    torch::jit::Parser parser(src);
                    auto tree = parser.parseFunction(false);
                } catch (...) {
                    // Parsing errors are expected for random input
                }
                break;
            }
            
            case 1: {
                // Compile a TorchScript function via Module::define
                try {
                    torch::jit::Module module("test_module");
                    module.define(script_content);
                    
                    // Try to run the compiled function with our tensor
                    std::vector<torch::jit::IValue> inputs;
                    inputs.push_back(input_tensor);
                    module.forward(inputs);
                } catch (...) {
                    // Compilation/execution errors expected for random input
                }
                break;
            }
            
            case 2: {
                // Test script module with registered parameter
                try {
                    torch::jit::Module module("param_module");
                    module.register_parameter("weight", input_tensor.clone(), false);
                    
                    // Create a method that uses the parameter
                    std::string method_def = "def forward(self, x):\n  return x + self.weight";
                    module.define(method_def);
                    
                    // Try to run the method
                    std::vector<torch::jit::IValue> inputs;
                    inputs.push_back(input_tensor);
                    auto output = module.forward(inputs);
                    
                    // Exercise the output
                    if (output.isTensor()) {
                        output.toTensor().sum();
                    }
                } catch (...) {
                    // Errors expected
                }
                break;
            }
            
            case 3: {
                // Test CompilationUnit with class definition
                try {
                    std::string class_script = R"(
class TestClass:
    value : int
    def __init__(self, v: int):
        self.value = v
    def get_value(self) -> int:
        return self.value
)";
                    torch::jit::CompilationUnit cu;
                    cu.define(c10::nullopt, class_script, torch::jit::nativeResolver(), nullptr);
                    
                    // Also try compiling the fuzz-generated content
                    try {
                        torch::jit::CompilationUnit cu2;
                        cu2.define(c10::nullopt, script_content, torch::jit::nativeResolver(), nullptr);
                    } catch (...) {
                        // Expected to fail for random input
                    }
                } catch (...) {
                    // Compilation errors expected
                }
                break;
            }
        }
        
        // Additional frontend exercising: try various valid script patterns
        // with fuzz-derived modifications
        try {
            // Create simple arithmetic scripts based on input data
            uint8_t op_byte = (offset < Size) ? Data[offset] : 0;
            std::string op;
            switch (op_byte % 6) {
                case 0: op = "+"; break;
                case 1: op = "-"; break;
                case 2: op = "*"; break;
                case 3: op = "/"; break;
                case 4: op = "**"; break;
                case 5: op = "%"; break;
            }
            
            std::string simple_script = "def compute(x, y):\n  return x " + op + " y";
            
            torch::jit::Module compute_module("compute_module");
            compute_module.define(simple_script);
            
            std::vector<torch::jit::IValue> compute_inputs;
            compute_inputs.push_back(input_tensor);
            compute_inputs.push_back(input_tensor);
            
            auto result = compute_module.get_method("compute")(compute_inputs);
            if (result.isTensor()) {
                result.toTensor().sum();
            }
        } catch (...) {
            // Division by zero, shape mismatch, etc. expected
        }
        
        // Test parsing expressions
        try {
            std::string expr_script = "def expr_test(a):\n  return " + script_content;
            auto src = std::make_shared<torch::jit::Source>(expr_script);
            torch::jit::Parser parser(src);
            parser.parseFunction(false);
        } catch (...) {
            // Parse errors expected
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}