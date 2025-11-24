#include "fuzzer_utils.h"   // General fuzzing utilities
#include <c10/util/Exception.h>
#include <iostream>         // For cerr
#include <tuple>            // For std::get with lu_unpack result
#include <vector>
#include <torch/script.h>

// Target API: torch.jit.ScriptWarning
// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor from the fuzzer data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple script module that will trigger a warning
        std::string script_code;
        
        // Use remaining bytes to determine which warning scenario to test
        if (offset < Size) {
            uint8_t scenario = Data[offset++] % 4;
            
            switch (scenario) {
                case 0:
                    // Unused variable warning
                    script_code = R"(
                        def forward(self, x):
                            unused_var = x + 1
                            return x
                    )";
                    break;
                    
                case 1:
                    // Deprecated feature warning
                    script_code = R"(
                        def forward(self, x):
                            a = []
                            a.append(x)
                            return a[0]
                    )";
                    break;
                    
                case 2:
                    // Type annotation warning
                    script_code = R"(
                        def forward(self, x):
                            y = x
                            return y
                    )";
                    break;
                    
                case 3:
                    // Potential undefined behavior
                    script_code = R"(
                        def forward(self, x):
                            if x.sum() > 0:
                                return x
                            return x
                    )";
                    break;
            }
        } else {
            // Default script if we don't have enough data
            script_code = R"(
                def forward(self, x):
                    unused = x * 2
                    return x
            )";
        }
        
        // Try to compile the script
        try {
            auto module = torch::jit::compile(script_code);
            
            // Try to run the module with our tensor
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(tensor);
            
            auto output = module->run_method("forward", tensor);
            
        } catch (const std::exception& e) {
            // JIT compilation errors are expected in some cases
            return 0;
        }
        
        // Test warning handling if possible
        if (offset < Size) {
            try {
                // Build a warning and dispatch through the warning utilities
                std::string warning_msg = "Test warning message";
                int line = Data[offset++] % 100;
                c10::Warning warning(
                    c10::Warning::UserWarning(),
                    c10::SourceLocation{__func__, __FILE__, static_cast<uint32_t>(line)},
                    warning_msg,
                    /*verbatim=*/true);
                c10::warn(warning);
                
            } catch (...) {
                // Ignore exceptions from warning handling
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
