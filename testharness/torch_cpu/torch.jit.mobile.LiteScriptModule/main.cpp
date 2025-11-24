#include "fuzzer_utils.h" // General fuzzing utilities
#include <torch/script.h>
#include <torch/csrc/jit/mobile/flatbuffer_loader.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/import.h>
#include <cstdio>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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

        // Script a simple module and save it in the mobile flatbuffer format
        torch::jit::Module scripted_module("lite_module");
        scripted_module.define(R"JIT(
            def forward(self, x):
                return x * 2.0 + 1.0
        )JIT");

        std::string temp_filename = "temp_mobile_model.ptl";
        scripted_module._save_for_mobile(temp_filename, {}, false, true);

        try {
            auto mobile_module = torch::jit::load_mobile_module_from_file(temp_filename);

            // Run forward and touch the result to exercise execution
            std::vector<c10::IValue> inputs;
            inputs.emplace_back(input_tensor);
            auto output = mobile_module.forward(inputs);
            if (output.isTensor()) {
                output.toTensor().sum();
            }

            // Touch additional APIs on LiteScriptModule
            for (const auto& method : mobile_module.get_methods()) {
                if (method.name() != "forward") {
                    auto maybe_output = method({input_tensor});
                    if (maybe_output.isTensor()) {
                        maybe_output.toTensor().sum();
                    }
                }
            }

            for (const auto& param : mobile_module.parameters()) {
                if (param.defined()) {
                    param.sum().item<double>();
                }
            }

            auto metadata = mobile_module.getMetadata();
            metadata["seen"] = "true";
            mobile_module.setMetadata(metadata);
            auto info = torch::jit::mobile::get_module_info(mobile_module);
            (void)info.bytecode_version;
        } catch (const std::exception &) {
            // Ignore failures from model loading/execution
        }

        std::remove(temp_filename.c_str());
    }
    catch (const std::exception &)
    {
        return -1; // discard the input
    }
    return 0; // keep the input
}
