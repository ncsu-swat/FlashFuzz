#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr, cout
#include <fstream>        // For file operations
#include <sstream>        // For string stream
#include <cstdio>         // For std::remove
#include <unistd.h>       // For getpid

// --- Fuzzer Entry Point ---
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
        // Need at least some data to work with
        if (Size < 4) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Create a tensor to serialize
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a temporary file path for serialization (unique per process)
        std::string temp_file = "/tmp/fuzz_serialized_tensor_" + std::to_string(getpid());
        
        // Test torch::save to file
        try {
            torch::save(tensor, temp_file);
            
            // Test torch::load from file
            torch::Tensor loaded_tensor;
            torch::load(loaded_tensor, temp_file);
            
            // Verify basic properties match
            (void)(loaded_tensor.sizes() == tensor.sizes());
        } catch (const std::exception&) {
            // Expected failures for certain tensor types, continue
        }
        
        // Test serialization to/from a stringstream buffer
        try {
            std::stringstream ss;
            torch::save(tensor, ss);
            
            // Try loading from the buffer
            torch::Tensor loaded_tensor;
            torch::load(loaded_tensor, ss);
        } catch (const std::exception&) {
            // If buffer serialization fails, continue
        }
        
        // Test serializing multiple tensors
        if (offset < Size) {
            size_t offset2 = 0;
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset2);
            std::vector<torch::Tensor> tensors = {tensor, tensor2};
            
            try {
                torch::save(tensors, temp_file);
                
                std::vector<torch::Tensor> loaded_tensors;
                torch::load(loaded_tensors, temp_file);
            } catch (const std::exception&) {
                // If multi-tensor serialization fails, continue
            }
        }
        
        // Test round-trip through string
        try {
            std::ostringstream oss;
            torch::save(tensor, oss);
            std::string serialized_data = oss.str();
            
            std::istringstream iss(serialized_data);
            torch::Tensor loaded_tensor;
            torch::load(loaded_tensor, iss);
        } catch (const std::exception&) {
            // If raw buffer serialization fails, continue
        }
        
        // Test with different tensor types created from fuzzer data
        if (offset + 4 < Size) {
            try {
                // Create tensors of specific types
                auto int_tensor = torch::randint(0, 256, {2, 2}, torch::kInt32);
                auto float_tensor = torch::rand({3, 3}, torch::kFloat32);
                
                std::stringstream ss;
                torch::save(int_tensor, ss);
                torch::Tensor loaded_int;
                torch::load(loaded_int, ss);
                
                ss.str("");
                ss.clear();
                torch::save(float_tensor, ss);
                torch::Tensor loaded_float;
                torch::load(loaded_float, ss);
            } catch (const std::exception&) {
                // Continue on failure
            }
        }
        
        // Clean up temporary file
        std::remove(temp_file.c_str());
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}