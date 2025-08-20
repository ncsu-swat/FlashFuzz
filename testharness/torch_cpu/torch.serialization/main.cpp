#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <fstream>        // For file operations
#include <sstream>        // For string stream

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to work with
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to serialize
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a temporary file path for serialization
        std::string temp_file = "temp_serialized_tensor";
        
        // Test torch::save
        try {
            torch::save(tensor, temp_file);
        } catch (const std::exception&) {
            // If save fails, try another approach
        }
        
        // Test torch::load
        try {
            torch::Tensor loaded_tensor;
            torch::load(loaded_tensor, temp_file);
        } catch (const std::exception&) {
            // If load fails, continue
        }
        
        // Test serialization to a buffer
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
        if (offset + 8 < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            std::vector<torch::Tensor> tensors = {tensor, tensor2};
            
            try {
                torch::save(tensors, temp_file);
                
                std::vector<torch::Tensor> loaded_tensors;
                torch::load(loaded_tensors, temp_file);
            } catch (const std::exception&) {
                // If multi-tensor serialization fails, continue
            }
        }
        
        // Test serializing to/from a raw buffer
        try {
            std::stringstream ss;
            torch::save(tensor, ss);
            std::string serialized_data = ss.str();
            
            std::istringstream input_stream(serialized_data);
            torch::Tensor loaded_tensor;
            torch::load(loaded_tensor, input_stream);
        } catch (const std::exception&) {
            // If raw buffer serialization fails, continue
        }
        
        // Clean up temporary file
        std::remove(temp_file.c_str());
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}