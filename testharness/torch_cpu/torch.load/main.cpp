#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <fstream>        // For file operations
#include <filesystem>     // For filesystem operations

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to work with
        if (Size < 4) {
            return 0;
        }
        
        // Create a temporary file to save the tensor
        std::string temp_filename = "temp_tensor_" + std::to_string(reinterpret_cast<uintptr_t>(Data)) + ".pt";
        
        // First, create a tensor to save
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Save the tensor to a file
        torch::save(tensor, temp_filename);
        
        // Try to load the tensor back
        torch::Tensor loaded_tensor;
        torch::load(loaded_tensor, temp_filename);
        
        // Try different loading options if there's more data
        if (offset + 1 < Size) {
            uint8_t options_byte = Data[offset++];
            
            // Try loading with different device options
            if (options_byte & 0x01) {
                try {
                    torch::Tensor device_tensor;
                    torch::load(device_tensor, temp_filename, torch::Device(torch::kCPU));
                } catch (...) {
                    // Ignore exceptions from device-specific loading
                }
            }
        }
        
        // Try loading from a non-existent file
        if (offset + 1 < Size) {
            try {
                torch::Tensor nonexistent_tensor;
                torch::load(nonexistent_tensor, "nonexistent_file_" + std::to_string(Data[offset]));
            } catch (...) {
                // Expected to fail, ignore
            }
        }
        
        // Try loading from a file with invalid content
        if (offset + 8 < Size) {
            std::string invalid_filename = "invalid_tensor_" + std::to_string(reinterpret_cast<uintptr_t>(Data)) + ".pt";
            std::ofstream invalid_file(invalid_filename, std::ios::binary);
            if (invalid_file) {
                invalid_file.write(reinterpret_cast<const char*>(Data + offset), Size - offset);
                invalid_file.close();
                
                try {
                    torch::Tensor invalid_tensor;
                    torch::load(invalid_tensor, invalid_filename);
                } catch (...) {
                    // Expected to fail, ignore
                }
                
                std::remove(invalid_filename.c_str());
            }
        }
        
        // Clean up the temporary file
        std::remove(temp_filename.c_str());
        
        // Try loading multiple tensors
        if (offset + 4 < Size) {
            std::string multi_filename = "multi_tensor_" + std::to_string(reinterpret_cast<uintptr_t>(Data)) + ".pt";
            
            // Create and save multiple tensors
            std::vector<torch::Tensor> tensors;
            for (int i = 0; i < 2 && offset < Size; i++) {
                try {
                    tensors.push_back(fuzzer_utils::createTensor(Data, Size, offset));
                } catch (...) {
                    break;
                }
            }
            
            if (!tensors.empty()) {
                torch::save(tensors, multi_filename);
                
                // Load multiple tensors
                std::vector<torch::Tensor> loaded_tensors;
                torch::load(loaded_tensors, multi_filename);
                
                std::remove(multi_filename.c_str());
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
