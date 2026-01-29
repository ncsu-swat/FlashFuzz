#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <fstream>        // For file operations
#include <cstdio>         // For std::remove

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }
    
    // Use fixed filenames to avoid file accumulation
    const std::string temp_filename = "/tmp/fuzz_tensor.pt";
    const std::string temp_filename2 = "/tmp/fuzz_tensors.pt";
    const std::string temp_archive = "/tmp/fuzz_tensor.archive";
    
    try
    {
        size_t offset = 0;
        
        // Skip if data is too small
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test torch::save functionality with single tensor
        torch::save(tensor, temp_filename);
        
        // Verify the save worked by loading the tensor back
        torch::Tensor loaded_tensor;
        torch::load(loaded_tensor, temp_filename);
        
        // Try to save multiple tensors in a vector
        if (Size > offset + 4) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            std::vector<torch::Tensor> tensors = {tensor, tensor2};
            
            torch::save(tensors, temp_filename2);
            
            // Load the tensors back
            std::vector<torch::Tensor> loaded_tensors;
            torch::load(loaded_tensors, temp_filename2);
            
            // Clean up
            std::remove(temp_filename2.c_str());
        }
        
        // Try to save using OutputArchive for more coverage
        if (Size > offset + 2) {
            torch::serialize::OutputArchive output_archive;
            output_archive.write("tensor", tensor);
            output_archive.save_to(temp_archive);
            
            // Verify by loading with InputArchive
            torch::serialize::InputArchive input_archive;
            input_archive.load_from(temp_archive);
            torch::Tensor archive_loaded;
            input_archive.read("tensor", archive_loaded);
            
            // Clean up archive file
            std::remove(temp_archive.c_str());
        }
        
        // Try saving tensors with different dtypes for coverage
        if (Size > offset + 1) {
            uint8_t dtype_byte = Data[offset++] % 4;
            torch::Tensor typed_tensor;
            
            switch (dtype_byte) {
                case 0:
                    typed_tensor = tensor.to(torch::kFloat32);
                    break;
                case 1:
                    typed_tensor = tensor.to(torch::kFloat64);
                    break;
                case 2:
                    typed_tensor = tensor.to(torch::kInt32);
                    break;
                case 3:
                    typed_tensor = tensor.to(torch::kInt64);
                    break;
            }
            
            torch::save(typed_tensor, temp_filename);
            torch::Tensor loaded_typed;
            torch::load(loaded_typed, temp_filename);
        }
        
        // Try to save to an invalid path (expected to fail)
        if (Size > offset + 1) {
            uint8_t dir_byte = Data[offset++];
            if (dir_byte % 10 == 0) {  // Occasionally try invalid paths
                try {
                    std::string invalid_path = "/nonexistent_dir_xyz/tensor.pt";
                    torch::save(tensor, invalid_path);
                } catch (...) {
                    // Expected to fail, continue silently
                }
            }
        }
        
        // Clean up the main temporary file
        std::remove(temp_filename.c_str());
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        // Clean up files on exception
        std::remove(temp_filename.c_str());
        std::remove(temp_filename2.c_str());
        std::remove(temp_archive.c_str());
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0;  // Keep the input
}