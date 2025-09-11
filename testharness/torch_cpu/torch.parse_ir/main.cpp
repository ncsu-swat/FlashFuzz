#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/csrc/jit/frontend/parser.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a meaningful test
        if (Size < 4) {
            return 0;
        }
        
        // Create a string from the input data to use as IR
        std::string ir_string;
        if (Size > 0) {
            // Use the data as a string, ensuring it's null-terminated
            size_t string_length = std::min(Size, static_cast<size_t>(1024)); // Limit string size
            ir_string = std::string(reinterpret_cast<const char*>(Data), string_length);
        } else {
            ir_string = "";
        }
        
        // Try to parse the IR string
        try {
            torch::jit::parseIR(ir_string, nullptr);
        } catch (...) {
            // Parsing errors are expected for random data, just continue
        }
        
        // Try with some additional context
        if (Size > 100) {
            try {
                // Create a simple module as context
                auto module = std::make_shared<torch::jit::Module>();
                torch::jit::parseIR(ir_string, module.get());
            } catch (...) {
                // Parsing errors are expected for random data, just continue
            }
        }
        
        // Try with different source name
        if (Size > 200) {
            try {
                std::string source_name = "test_source";
                if (Size > 220) {
                    source_name = std::string(reinterpret_cast<const char*>(Data + 200), 
                                             std::min(Size - 200, static_cast<size_t>(20)));
                }
                torch::jit::parseIR(ir_string, nullptr, source_name);
            } catch (...) {
                // Parsing errors are expected for random data, just continue
            }
        }
        
        // Try with a tensor as input
        if (Size > 300) {
            try {
                // Create a tensor from the remaining data
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Convert tensor to string representation and try to parse it
                std::stringstream ss;
                ss << tensor;
                std::string tensor_str = ss.str();
                
                torch::jit::parseIR(tensor_str, nullptr);
            } catch (...) {
                // Errors are expected, just continue
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
