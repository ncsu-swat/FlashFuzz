#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/parse_operators.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a meaningful test
        if (Size < 4) {
            return 0;
        }
        
        // Create a string from the input data to test as a map_location parameter
        std::string map_location;
        size_t str_len = std::min(Size - offset, static_cast<size_t>(32));
        if (str_len > 0) {
            map_location = std::string(reinterpret_cast<const char*>(Data + offset), str_len);
            offset += str_len;
        }
        
        // Create a few different device strings to test
        std::vector<std::string> device_strings = {
            map_location,
            "cpu",
            "cuda",
            "cuda:0",
            "cuda:1",
            "mps",
            "xla",
            "vulkan",
            "",
            "invalid_device",
            "cpu cpu",
            "cuda:0 cpu",
            "cpu cuda:0"
        };
        
        // Test validate_map_location with different inputs
        for (const auto& device : device_strings) {
            try {
                torch::jit::mobile::validate_map_location(device);
            } catch (const c10::Error& e) {
                // Expected exceptions from PyTorch are fine
            }
        }
        
        // Test with additional inputs derived from the fuzzer data
        if (Size - offset >= 2) {
            uint8_t byte1 = Data[offset++];
            uint8_t byte2 = Data[offset++];
            
            // Create a device string with potential special characters
            std::string custom_device;
            for (size_t i = 0; i < byte1 % 10 && offset < Size; i++) {
                custom_device += static_cast<char>(Data[offset++]);
            }
            
            // Try to validate the custom device string
            try {
                torch::jit::mobile::validate_map_location(custom_device);
            } catch (const c10::Error& e) {
                // Expected exceptions from PyTorch are fine
            }
            
            // Create a more complex map_location string with device mappings
            if (byte2 % 3 == 0 && offset < Size - 10) {
                std::string src_device = "cuda:" + std::to_string(byte2 % 8);
                std::string dst_device = "cuda:" + std::to_string((byte2 + 1) % 8);
                std::string mapping = src_device + " " + dst_device;
                
                try {
                    torch::jit::mobile::validate_map_location(mapping);
                } catch (const c10::Error& e) {
                    // Expected exceptions from PyTorch are fine
                }
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