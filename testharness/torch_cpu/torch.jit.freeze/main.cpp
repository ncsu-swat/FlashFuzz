#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        torch::jit::Module module("TestModule");
        
        auto forward_func = [&input_tensor](const torch::jit::Stack& stack) -> torch::jit::Stack {
            torch::jit::Stack output = stack;
            output.emplace_back(input_tensor);
            return output;
        };
        
        module.define(R"(
            def forward(self, x):
                return x
        )");
        
        module.register_parameter("weight", input_tensor, false);
        
        torch::jit::script::Module script_module(module);
        
        auto graph = script_module.get_method("forward").graph();
        
        bool preserve_parameters = false;
        if (offset < Size) {
            preserve_parameters = Data[offset++] & 0x1;
        }
        
        bool disable_shape_peephole = false;
        if (offset < Size) {
            disable_shape_peephole = Data[offset++] & 0x1;
        }
        
        try {
            torch::jit::freeze(script_module, preserve_parameters ? c10::optional<std::vector<std::string>>{} : c10::nullopt);
        } catch (const c10::Error& e) {
            return 0;
        }
        
        try {
            auto result = script_module.forward({input_tensor});
        } catch (const c10::Error& e) {
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