#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <fstream>        // For file operations

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a meaningful test
        if (Size < 4) {
            return 0;
        }
        
        // Create a temporary file to test from_file functionality
        std::string temp_filename = "temp_fuzzer_file";
        
        // Create a tensor to write to the file
        torch::Tensor tensor;
        
        // Decide what kind of data to write to the file
        if (offset < Size) {
            uint8_t file_content_type = Data[offset++];
            
            if (file_content_type % 3 == 0) {
                // Create a tensor from the fuzzer data and save it
                tensor = fuzzer_utils::createTensor(Data, Size, offset);
                torch::save(tensor, temp_filename);
            } else if (file_content_type % 3 == 1) {
                // Write raw binary data to the file
                std::ofstream file(temp_filename, std::ios::binary);
                if (file) {
                    size_t bytes_to_write = std::min(Size - offset, static_cast<size_t>(1024));
                    file.write(reinterpret_cast<const char*>(Data + offset), bytes_to_write);
                    offset += bytes_to_write;
                    file.close();
                }
            } else {
                // Write a specific tensor format
                torch::Tensor specific_tensor;
                
                // Choose tensor type based on next byte
                if (offset < Size) {
                    uint8_t tensor_type = Data[offset++];
                    
                    if (tensor_type % 5 == 0) {
                        // Empty tensor
                        specific_tensor = torch::empty({0});
                    } else if (tensor_type % 5 == 1) {
                        // Scalar tensor
                        specific_tensor = torch::tensor(3.14);
                    } else if (tensor_type % 5 == 2) {
                        // 1D tensor
                        specific_tensor = torch::ones({5});
                    } else if (tensor_type % 5 == 3) {
                        // 2D tensor
                        specific_tensor = torch::eye(3);
                    } else {
                        // 3D tensor
                        specific_tensor = torch::ones({2, 3, 4});
                    }
                } else {
                    specific_tensor = torch::ones({1});
                }
                
                torch::save(specific_tensor, temp_filename);
            }
        }
        
        // Test from_file functionality with different options
        if (offset < Size) {
            uint8_t option_byte = Data[offset++];
            
            // Test different combinations of options
            bool shared = (option_byte & 0x01) != 0;
            bool requires_grad = (option_byte & 0x02) != 0;
            
            // Try to load the tensor from file
            torch::Tensor loaded_tensor;
            
            try {
                // Test from_file with different options
                if (option_byte % 4 == 0) {
                    // Basic usage
                    loaded_tensor = torch::from_file(temp_filename, shared);
                } else if (option_byte % 4 == 1) {
                    // With specific size
                    int64_t size_value = 1;
                    if (offset + sizeof(int64_t) <= Size) {
                        std::memcpy(&size_value, Data + offset, sizeof(int64_t));
                        offset += sizeof(int64_t);
                    }
                    size_value = std::abs(size_value) % 1000 + 1; // Ensure positive and reasonable
                    
                    loaded_tensor = torch::from_file(temp_filename, shared, size_value);
                } else if (option_byte % 4 == 2) {
                    // With dtype
                    torch::ScalarType dtype = fuzzer_utils::parseDataType(option_byte);
                    loaded_tensor = torch::from_file(temp_filename, 
                                                    shared,
                                                    std::nullopt,
                                                    torch::TensorOptions()
                                                        .dtype(dtype)
                                                        .requires_grad(requires_grad));
                } else {
                    // With non-existent file
                    loaded_tensor = torch::from_file("nonexistent_file_" + std::to_string(option_byte), shared);
                }
                
                // Perform some operations on the loaded tensor to ensure it's valid
                if (loaded_tensor.defined()) {
                    auto tensor_sum = loaded_tensor.sum();
                    auto tensor_mean = loaded_tensor.mean();
                    
                    // Try to modify the tensor
                    if (loaded_tensor.is_floating_point() && loaded_tensor.numel() > 0) {
                        loaded_tensor[0] = 42.0;
                    }
                    
                    // Try to reshape if possible
                    if (loaded_tensor.numel() > 1) {
                        auto new_shape = loaded_tensor.sizes().vec();
                        if (new_shape.size() > 1) {
                            std::swap(new_shape[0], new_shape[new_shape.size() - 1]);
                            try {
                                auto reshaped = loaded_tensor.reshape(new_shape);
                            } catch (...) {
                                // Reshape might fail, that's okay
                            }
                        }
                    }
                }
            } catch (...) {
                // Exceptions from from_file are expected in some cases
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
