#include "fuzzer_utils.h"
#include <torch/script.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/import.h>
#include <cstdio>
#include <sstream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    const char *keyword = "torch.jit.mobile.LiteScriptModule";
    (void)keyword;

    try
    {
        size_t offset = 0;

        if (Size < 4) {
            return 0;
        }

        // Create a bounded tensor to use as input for our module
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset).to(torch::kFloat);
        
        // Ensure input tensor has reasonable shape for forward pass
        if (input_tensor.numel() == 0) {
            input_tensor = torch::randn({1, 4});
        }

        // Script a simple module and save it in the mobile flatbuffer format
        torch::jit::Module scripted_module("lite_module");
        scripted_module.define(R"JIT(
            def forward(self, x):
                return x * 2.0 + 1.0
        )JIT");

        // Use a unique temp filename to avoid race conditions
        std::ostringstream filename_ss;
        filename_ss << "/tmp/temp_mobile_model_" << iteration_count << ".ptl";
        std::string temp_filename = filename_ss.str();

        try {
            scripted_module._save_for_mobile(temp_filename, {}, false, true);
        } catch (const std::exception &) {
            // _save_for_mobile may fail, ignore
            return 0;
        }

        try {
            // Load the mobile module - this is the main API we're testing
            torch::jit::mobile::Module mobile_module = torch::jit::_load_for_mobile(temp_filename);

            // Run forward and touch the result to exercise execution
            std::vector<c10::IValue> inputs;
            inputs.emplace_back(input_tensor);
            
            try {
                auto output = mobile_module.forward(inputs);
                if (output.isTensor()) {
                    auto result = output.toTensor();
                    result.sum().item<float>();
                }
            } catch (const std::exception &) {
                // Forward may fail due to shape issues, ignore
            }

            // Exercise get_methods if available
            try {
                auto methods = mobile_module.get_methods();
                for (const auto& method : methods) {
                    (void)method.name();
                }
            } catch (const std::exception &) {
                // Method enumeration may fail
            }

            // Exercise parameters access
            try {
                auto params = mobile_module.parameters();
                for (const auto& param : params) {
                    if (param.defined()) {
                        param.sum().item<double>();
                    }
                }
            } catch (const std::exception &) {
                // Parameter access may fail
            }

            // Exercise named_parameters - returns pairs of (string, Tensor)
            try {
                auto named_params = mobile_module.named_parameters();
                for (const auto& named_param : named_params) {
                    (void)named_param.first;  // name is first element of pair
                    if (named_param.second.defined()) {  // tensor is second element
                        named_param.second.numel();
                    }
                }
            } catch (const std::exception &) {
                // Named parameter access may fail
            }

            // Exercise run_method
            try {
                auto output2 = mobile_module.run_method("forward", input_tensor);
                if (output2.isTensor()) {
                    output2.toTensor().numel();
                }
            } catch (const std::exception &) {
                // run_method may fail
            }

            // Exercise find_method
            try {
                auto method_ptr = mobile_module.find_method("forward");
                if (method_ptr) {
                    (void)method_ptr->name();
                }
            } catch (const std::exception &) {
                // find_method may fail
            }

            // Exercise get_class_type
            try {
                auto class_type = mobile_module.type();
                if (class_type) {
                    (void)class_type->name();
                }
            } catch (const std::exception &) {
                // type access may fail
            }

            // Exercise attr access with a known attribute name
            try {
                // Check if module has any attributes via hasattr-like pattern
                auto ivalue = mobile_module.attr("training", c10::IValue());
                (void)ivalue;
            } catch (const std::exception &) {
                // attr access may fail
            }

        } catch (const std::exception &) {
            // Model loading may fail, this is expected for some inputs
        }

        // Clean up temp file
        std::remove(temp_filename.c_str());
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}