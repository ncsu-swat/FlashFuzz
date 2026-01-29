#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create two input tensors for outer product
        torch::Tensor vec1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset >= Size) {
            // Try to use the operation with just one tensor (self outer product)
            try {
                torch::Tensor flat_vec = vec1.flatten();
                auto result = torch::outer(flat_vec, flat_vec);
            } catch (...) {
                // Expected to fail in some cases
            }
            return 0;
        }
        
        torch::Tensor vec2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Flatten tensors to 1D as required by torch::outer
        if (vec1.dim() != 1) {
            vec1 = vec1.flatten();
        }
        
        if (vec2.dim() != 1) {
            vec2 = vec2.flatten();
        }
        
        // Apply the outer product operation
        auto result = torch::outer(vec1, vec2);
        
        // Verify result shape is correct: [vec1.size(0), vec2.size(0)]
        if (result.dim() != 2 || result.size(0) != vec1.size(0) || result.size(1) != vec2.size(0)) {
            std::cerr << "Unexpected result shape" << std::endl;
        }
        
        // Test with empty tensors
        if (offset + 1 < Size) {
            uint8_t test_empty = Data[offset++];
            if (test_empty % 4 == 0) {
                // Test with empty first vector
                torch::Tensor empty_vec1 = torch::empty({0}, vec1.options());
                try {
                    auto result_empty1 = torch::outer(empty_vec1, vec2);
                } catch (...) {
                    // Expected to fail in some cases
                }
            } else if (test_empty % 4 == 1) {
                // Test with empty second vector
                torch::Tensor empty_vec2 = torch::empty({0}, vec2.options());
                try {
                    auto result_empty2 = torch::outer(vec1, empty_vec2);
                } catch (...) {
                    // Expected to fail in some cases
                }
            } else if (test_empty % 4 == 2) {
                // Test with both empty vectors
                torch::Tensor empty_vec1 = torch::empty({0}, vec1.options());
                torch::Tensor empty_vec2 = torch::empty({0}, vec2.options());
                try {
                    auto result_empty_both = torch::outer(empty_vec1, empty_vec2);
                } catch (...) {
                    // Expected to fail in some cases
                }
            }
        }
        
        // Test with different data types if we have more data
        if (offset + 1 < Size) {
            uint8_t test_dtype = Data[offset++];
            if (test_dtype % 3 == 0 && offset < Size) {
                // Try converting to a different dtype
                torch::ScalarType target_dtype = fuzzer_utils::parseDataType(Data[offset++]);
                try {
                    auto converted_vec1 = vec1.to(target_dtype);
                    auto converted_vec2 = vec2.to(target_dtype);
                    auto result_converted = torch::outer(converted_vec1, converted_vec2);
                } catch (...) {
                    // Some dtype conversions might fail
                }
            }
        }
        
        // Test with out parameter if we have more data
        if (offset + 1 < Size) {
            uint8_t test_out = Data[offset++];
            if (test_out % 2 == 0 && vec1.size(0) > 0 && vec2.size(0) > 0) {
                try {
                    // Create output tensor with correct shape
                    auto out_tensor = torch::empty({vec1.size(0), vec2.size(0)}, result.options());
                    torch::outer_outf(vec1, vec2, out_tensor);
                } catch (...) {
                    // Might fail for some combinations
                }
            }
        }
        
        // Test ger (alias for outer) if we have more data
        if (offset + 1 < Size) {
            uint8_t test_ger = Data[offset++];
            if (test_ger % 2 == 0) {
                try {
                    auto result_ger = torch::ger(vec1, vec2);
                } catch (...) {
                    // Might fail
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}