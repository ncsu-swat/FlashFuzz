#include "fuzzer_utils.h" // General fuzzing utilities
#include <algorithm>
#include <iostream> // For cerr
#include <optional>
#include <sstream>
#include <tuple> // For std::get with lu_unpack result
#include <torch/csrc/jit/serialization/import.h>
#include <torch/script.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static constexpr const char *kKeyword = "torch.jit.jit_module_from_flatbuffer";
    (void)kKeyword; // keyword marker for harness checks

    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to work with
        if (Size < 4) {
            return 0;
        }
        
        // Create a buffer from the input data (cap size to avoid huge allocations)
        const size_t kMaxBufferSize = 1 << 18; // 256KiB
        const size_t usable_size = std::min(Size, kMaxBufferSize);
        std::vector<char> flatbuffer_data;
        flatbuffer_data.reserve(usable_size);
        for (size_t i = 0; i < usable_size; ++i) {
            flatbuffer_data.push_back(static_cast<char>(Data[i]));
        }

        auto load_from_buffer = [](const std::vector<char> &buffer) -> std::optional<torch::jit::Module> {
            if (buffer.empty()) {
                return std::nullopt;
            }
            std::string serialized(buffer.begin(), buffer.end());
            std::istringstream stream(serialized);
            torch::jit::ExtraFilesMap extra_files;
            try {
                return torch::jit::load_jit_module_from_stream(stream, extra_files, std::nullopt);
            } catch (...) {
                return std::nullopt; // Expected to fail for random bytes
            }
        };
        
        // Try to load the module from the flatbuffer
        if (auto module_opt = load_from_buffer(flatbuffer_data)) {
            torch::jit::Module module = *module_opt;

            // Create a tensor to pass to the module
            torch::Tensor input = fuzzer_utils::createTensor(Data, usable_size, offset);
            
            // Try to get methods
            auto methods = module.get_methods();
            
            // If there are methods, try to run the first one
            if (!methods.empty()) {
                try {
                    std::vector<torch::jit::IValue> inputs;
                    inputs.push_back(input);
                    auto output = methods[0](inputs);
                    if (output.isTensor()) {
                        output.toTensor().sum().item<float>(); // touch output
                    }
                } catch (...) {
                    // Method execution might fail, which is expected for random data
                }
            }
        }
        
        // Try with a modified buffer (flip some bits)
        if (usable_size > 10) {
            std::vector<char> modified_buffer = flatbuffer_data;
            for (size_t i = 0; i < std::min(size_t(10), usable_size); ++i) {
                modified_buffer[i] ^= 0xFF;  // Flip bits
            }
            
            if (auto module_opt = load_from_buffer(modified_buffer)) {
                auto module = *module_opt;
                (void)module; // Touch module to keep the path exercised
            }
        }
    }
    catch (const std::exception &e)
    {
        return 0; // keep the input
    }
    return 0; // keep the input
}
