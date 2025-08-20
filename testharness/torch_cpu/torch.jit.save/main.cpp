#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <fstream>        // For file operations
#include <string>         // For string operations
#include <torch/script.h> // For torch::jit functions

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple module to save
        torch::jit::Module module("test_module");
        
        // Register the tensor as a parameter in the module
        module.register_parameter("test_param", tensor, false);
        
        // Create a temporary filename for saving
        std::string filename = "/tmp/jit_save_fuzzer_" + std::to_string(reinterpret_cast<uintptr_t>(Data));
        
        // Try different save options based on remaining data
        if (offset < Size) {
            uint8_t save_option = Data[offset++] % 3;
            
            switch (save_option) {
                case 0:
                    // Standard save
                    module.save(filename);
                    break;
                    
                case 1:
                    // Save with extra files map
                    {
                        std::unordered_map<std::string, std::string> extra_files;
                        if (offset < Size - 1) {
                            std::string key = "extra_key_" + std::to_string(Data[offset++]);
                            std::string value = "extra_value_" + std::to_string(Data[offset++]);
                            extra_files[key] = value;
                        }
                        module.save(filename, extra_files);
                    }
                    break;
                    
                case 2:
                    // Save to output stream
                    {
                        std::ofstream output_file(filename, std::ios::binary);
                        if (output_file) {
                            if (offset < Size - 1) {
                                uint8_t stream_option = Data[offset++] % 2;
                                if (stream_option == 0) {
                                    module.save(output_file);
                                } else {
                                    std::unordered_map<std::string, std::string> extra_files;
                                    if (offset < Size - 1) {
                                        std::string key = "stream_key_" + std::to_string(Data[offset++]);
                                        std::string value = "stream_value_" + std::to_string(Data[offset++]);
                                        extra_files[key] = value;
                                    }
                                    module.save(output_file, extra_files);
                                }
                            } else {
                                module.save(output_file);
                            }
                            output_file.close();
                        }
                    }
                    break;
            }
        } else {
            // Default save if no more data
            module.save(filename);
        }
        
        // Try to load the saved module to verify it works
        try {
            torch::jit::Module loaded_module = torch::jit::load(filename);
        } catch (...) {
            // Ignore load errors - we're testing save functionality
        }
        
        // Clean up the temporary file
        std::remove(filename.c_str());
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}