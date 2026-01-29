#include "fuzzer_utils.h"
#include <iostream>
#include <fstream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Need at least a few bytes for meaningful testing
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Create a temporary file with raw binary data
        // Use unique filename based on iteration to avoid race conditions
        std::string temp_filename = "/tmp/fuzzer_from_file_" + std::to_string(iteration_count);

        // Write raw binary data to the file
        size_t bytes_to_write = std::min(Size - offset - 2, static_cast<size_t>(4096));
        if (bytes_to_write < 4) {
            return 0;
        }

        {
            std::ofstream file(temp_filename, std::ios::binary);
            if (!file) {
                return 0;
            }
            file.write(reinterpret_cast<const char*>(Data + offset), bytes_to_write);
            file.close();
        }
        offset += bytes_to_write;

        // Parse options from remaining fuzzer data
        uint8_t option_byte = 0;
        if (offset < Size) {
            option_byte = Data[offset++];
        }

        uint8_t dtype_byte = 0;
        if (offset < Size) {
            dtype_byte = Data[offset++];
        }

        bool shared = (option_byte & 0x01) != 0;

        // Determine dtype and element size
        torch::ScalarType dtype;
        size_t element_size;
        
        switch (dtype_byte % 6) {
            case 0:
                dtype = torch::kFloat32;
                element_size = 4;
                break;
            case 1:
                dtype = torch::kFloat64;
                element_size = 8;
                break;
            case 2:
                dtype = torch::kInt32;
                element_size = 4;
                break;
            case 3:
                dtype = torch::kInt64;
                element_size = 8;
                break;
            case 4:
                dtype = torch::kInt16;
                element_size = 2;
                break;
            default:
                dtype = torch::kUInt8;
                element_size = 1;
                break;
        }

        // Calculate how many elements we can read from the file
        int64_t num_elements = static_cast<int64_t>(bytes_to_write / element_size);
        if (num_elements < 1) {
            num_elements = 1;
        }

        // Clamp to reasonable size
        num_elements = std::min(num_elements, static_cast<int64_t>(1024));

        try {
            // Test from_file with different parameter combinations
            torch::Tensor loaded_tensor;

            auto options = torch::TensorOptions().dtype(dtype);

            if ((option_byte >> 1) % 3 == 0) {
                // With explicit size
                loaded_tensor = torch::from_file(temp_filename, shared, num_elements, options);
            } else if ((option_byte >> 1) % 3 == 1) {
                // With size = 0 (should read entire file)
                loaded_tensor = torch::from_file(temp_filename, shared, 0, options);
            } else {
                // With nullopt size
                loaded_tensor = torch::from_file(temp_filename, shared, c10::nullopt, options);
            }

            // Perform operations on the loaded tensor to exercise it
            if (loaded_tensor.defined() && loaded_tensor.numel() > 0) {
                // Basic operations
                auto numel = loaded_tensor.numel();
                auto sizes = loaded_tensor.sizes();
                auto dtype_check = loaded_tensor.dtype();

                // Sum (works on all numeric types)
                try {
                    auto tensor_sum = loaded_tensor.sum();
                } catch (...) {
                    // Some operations may fail on certain dtypes
                }

                // Clone and contiguous checks
                auto cloned = loaded_tensor.clone();
                auto is_contiguous = loaded_tensor.is_contiguous();

                // Try reshape if we have multiple elements
                if (numel > 1) {
                    try {
                        auto reshaped = loaded_tensor.reshape({-1});
                    } catch (...) {
                        // Reshape might fail
                    }
                }

                // Test shared memory behavior if shared=true
                if (shared && loaded_tensor.is_floating_point() && numel > 0) {
                    try {
                        // Modifying shared tensor should affect the file
                        loaded_tensor.index_put_({0}, 123.0);
                    } catch (...) {
                        // Modification might fail
                    }
                }

                // Test view operations
                try {
                    auto viewed = loaded_tensor.view({-1});
                } catch (...) {
                    // View might fail
                }
            }
        } catch (...) {
            // Exceptions from from_file are expected for invalid files/sizes
        }

        // Clean up the temporary file
        std::remove(temp_filename.c_str());
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}