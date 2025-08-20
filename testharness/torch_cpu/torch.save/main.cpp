#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <fstream>        // For file operations
#include <filesystem>     // For filesystem operations

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if data is too small
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a temporary filename for saving
        std::string temp_filename = "temp_tensor_" + std::to_string(reinterpret_cast<uintptr_t>(Data)) + ".pt";
        
        // Test torch::save functionality
        torch::save(tensor, temp_filename);
        
        // Verify the save worked by loading the tensor back
        torch::Tensor loaded_tensor;
        torch::load(loaded_tensor, temp_filename);
        
        // Try to save multiple tensors in a vector
        if (Size > offset + 4) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            std::vector<torch::Tensor> tensors = {tensor, tensor2};
            
            std::string temp_filename2 = "temp_tensors_" + std::to_string(reinterpret_cast<uintptr_t>(Data)) + ".pt";
            torch::save(tensors, temp_filename2);
            
            // Load the tensors back
            std::vector<torch::Tensor> loaded_tensors;
            torch::load(loaded_tensors, temp_filename2);
            
            // Clean up
            std::remove(temp_filename2.c_str());
        }
        
        // Try to save a tensor with options
        if (Size > offset + 2) {
            uint8_t option_byte = Data[offset++];
            
            torch::serialize::OutputArchive archive;
            archive.write("tensor", tensor);
            archive.save_to(temp_filename + ".archive");
        }
        
        // Try to save to a non-existent directory
        if (Size > offset + 2) {
            uint8_t dir_byte = Data[offset++];
            if (dir_byte % 5 == 0) {  // Occasionally try invalid paths
                try {
                    std::string invalid_path = "/nonexistent_dir_" + std::to_string(dir_byte) + "/tensor.pt";
                    torch::save(tensor, invalid_path);
                } catch (...) {
                    // Expected to fail, continue
                }
            }
        }
        
        // Clean up the temporary file
        std::remove(temp_filename.c_str());
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}