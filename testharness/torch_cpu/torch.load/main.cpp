#include "fuzzer_utils.h"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdint>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to work with
        if (Size < 4) {
            return 0;
        }
        
        // Create unique temporary filenames using iteration count and thread-safe naming
        std::string temp_filename = "/tmp/fuzz_tensor_" + std::to_string(iteration_count) + ".pt";
        std::string invalid_filename = "/tmp/fuzz_invalid_" + std::to_string(iteration_count) + ".pt";
        std::string multi_filename = "/tmp/fuzz_multi_" + std::to_string(iteration_count) + ".pt";
        
        // Create a tensor to save
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Save the tensor to a file
        torch::save(tensor, temp_filename);
        
        // Load the tensor back
        torch::Tensor loaded_tensor;
        torch::load(loaded_tensor, temp_filename);
        
        // Try loading with device option
        if (offset + 1 < Size) {
            uint8_t options_byte = Data[offset++];
            
            if (options_byte & 0x01) {
                try {
                    torch::Tensor device_tensor;
                    torch::load(device_tensor, temp_filename, torch::Device(torch::kCPU));
                } catch (...) {
                    // Expected failure for some inputs
                }
            }
        }
        
        // Try loading from a file with fuzz data as content (tests deserialization robustness)
        if (offset + 8 < Size) {
            std::ofstream invalid_file(invalid_filename, std::ios::binary);
            if (invalid_file) {
                invalid_file.write(reinterpret_cast<const char*>(Data + offset), Size - offset);
                invalid_file.close();
                
                try {
                    torch::Tensor invalid_tensor;
                    torch::load(invalid_tensor, invalid_filename);
                } catch (...) {
                    // Expected to fail with invalid data
                }
                
                std::remove(invalid_filename.c_str());
            }
        }
        
        // Try loading multiple tensors
        if (offset + 4 < Size) {
            std::vector<torch::Tensor> tensors;
            for (int i = 0; i < 3 && offset < Size; i++) {
                try {
                    tensors.push_back(fuzzer_utils::createTensor(Data, Size, offset));
                } catch (...) {
                    break;
                }
            }
            
            if (!tensors.empty()) {
                torch::save(tensors, multi_filename);
                
                std::vector<torch::Tensor> loaded_tensors;
                torch::load(loaded_tensors, multi_filename);
                
                std::remove(multi_filename.c_str());
            }
        }
        
        // Clean up main temporary file
        std::remove(temp_filename.c_str());
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}