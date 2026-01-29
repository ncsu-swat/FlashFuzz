#include "fuzzer_utils.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include <torch/script.h>
#include <torch/csrc/jit/serialization/import.h>

// Target API: torch.import_ir_module_from_buffer
// C++ equivalent: torch::jit::load with istream or torch::jit::import_ir_module

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Use first byte to determine which import path to try
        uint8_t import_mode = Data[offset++] % 3;

        // Create buffer from fuzzer data
        size_t buffer_size = std::min(static_cast<size_t>(4096), Size - offset);
        std::string buffer_str(reinterpret_cast<const char*>(Data + offset), buffer_size);
        offset += buffer_size;

        // Try different import methods based on mode
        if (import_mode == 0) {
            // Method 1: torch::jit::load with istream (equivalent to import_ir_module_from_buffer)
            try {
                std::istringstream buffer_stream(buffer_str);
                torch::jit::Module module = torch::jit::load(buffer_stream);
                
                // Try to get method names
                try {
                    for (const auto& method : module.get_methods()) {
                        (void)method.name();
                    }
                } catch (...) {
                    // Silently ignore
                }
            } catch (...) {
                // Import failure expected for random data
            }
        } else if (import_mode == 1) {
            // Method 2: import_ir_module with CompilationUnit
            try {
                auto cu = std::make_shared<torch::jit::CompilationUnit>();
                std::istringstream buffer_stream(buffer_str);
                torch::jit::Module module = torch::jit::import_ir_module(cu, buffer_stream);
                
                // Try to inspect the module
                try {
                    (void)module.is_training();
                } catch (...) {
                    // Silently ignore
                }
            } catch (...) {
                // Import failure expected for random data
            }
        } else {
            // Method 3: Try with device specification
            try {
                std::istringstream buffer_stream(buffer_str);
                torch::jit::Module module = torch::jit::load(buffer_stream, torch::kCPU);
                
                // Try to set to eval mode
                try {
                    module.eval();
                } catch (...) {
                    // Silently ignore
                }
            } catch (...) {
                // Import failure expected for random data
            }
        }

        // Also try parsing as extra files map (another import variant)
        try {
            std::istringstream buffer_stream(buffer_str);
            std::unordered_map<std::string, std::string> extra_files;
            extra_files["extra_data"] = "";  // Request extra file if present
            torch::jit::Module module = torch::jit::load(buffer_stream, torch::kCPU, extra_files);
        } catch (...) {
            // Expected to fail with random data
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}