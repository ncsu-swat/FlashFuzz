#include "fuzzer_utils.h" // General fuzzing utilities
#include <torch/script.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <cstdio>
#include <sstream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    const char *keyword = "torch.jit.mobile.torch";
    (void)keyword;
    try
    {
        size_t offset = 0;

        if (Size < 4) {
            return 0;
        }

        // Build a bounded tensor to exercise mobile execution paths.
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset).to(torch::kFloat);
        input_tensor = input_tensor.contiguous();

        // Script a tiny module we can serialize for mobile.
        torch::jit::Module module("mobile_module");
        module.define(R"JIT(
            def forward(self, x):
                return x * 2.0 + 1.0
        )JIT");

        // Save/load through a temporary file using the mobile loader.
        const bool use_flatbuffer = (Data[0] & 1) != 0;
        const std::string filename = "temp_mobile_model.ptl";
        module._save_for_mobile(filename, {}, /*save_mobile_debug_info=*/false, use_flatbuffer);

        try {
            auto loaded_file = torch::jit::_load_for_mobile(filename);
            std::vector<c10::IValue> inputs;
            inputs.emplace_back(input_tensor);
            auto file_output = loaded_file.forward(inputs);
            if (file_output.isTensor()) {
                file_output.toTensor().sum();
            }

            auto op_names = torch::jit::mobile::_export_operator_list(loaded_file);
            if (!op_names.empty()) {
                (void)*op_names.begin();
            }
        } catch (const std::exception &) {
            // Ignore loader/runtime failures; keep fuzzing.
        }

        // Save/load via a stream to cover the alternate API surface.
        try {
            std::stringstream ss;
            module._save_for_mobile(ss, {}, /*save_mobile_debug_info=*/false, use_flatbuffer);
            ss.seekg(0);
            auto loaded_stream = torch::jit::_load_for_mobile(ss);
            auto stream_output = loaded_stream.forward({input_tensor});
            if (stream_output.isTensor()) {
                stream_output.toTensor().sum();
            }
        } catch (const std::exception &) {
            // Ignore loader/runtime failures; keep fuzzing.
        }

        std::remove(filename.c_str());
    }
    catch (const std::exception &)
    {
        return -1; // discard the input
    }
    return 0; // keep the input
}
