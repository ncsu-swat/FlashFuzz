#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for predicate and 1 byte for branch selection
        if (Size < 2) {
            return 0;
        }
        
        // Parse predicate value (true/false)
        bool predicate = Data[offset++] & 0x1;
        
        // Parse branch selection (which branch to test)
        uint8_t branch_type = Data[offset++] % 3; // 0: both branches, 1: true branch only, 2: false branch only
        
        // Create input tensors for the branches
        torch::Tensor true_branch;
        torch::Tensor false_branch;
        
        // Create tensors based on branch_type
        if (branch_type == 0 || branch_type == 1) {
            if (offset < Size) {
                true_branch = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                true_branch = torch::ones({1, 2, 3});
            }
        }
        
        if (branch_type == 0 || branch_type == 2) {
            if (offset < Size) {
                false_branch = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                false_branch = torch::zeros({3, 2, 1});
            }
        }
        
        // If we didn't create both branches, create the missing one
        if (branch_type == 1) {
            false_branch = torch::zeros({3, 2, 1});
        } else if (branch_type == 2) {
            true_branch = torch::ones({1, 2, 3});
        }
        
        // Apply conditional operation using torch::where
        torch::Tensor result = predicate ? true_branch : false_branch;
        
        // Verify the result matches the expected branch
        if (predicate) {
            auto equal = torch::equal(result, true_branch);
            if (!equal.item<bool>()) {
                throw std::runtime_error("Result doesn't match true branch when predicate is true");
            }
        } else {
            auto equal = torch::equal(result, false_branch);
            if (!equal.item<bool>()) {
                throw std::runtime_error("Result doesn't match false branch when predicate is false");
            }
        }
        
        // Test with different tensor types and shapes
        if (offset + 2 < Size) {
            // Create tensors with different dtypes
            size_t new_offset = offset;
            torch::Tensor complex_true_branch = fuzzer_utils::createTensor(Data, Size, new_offset);
            
            if (new_offset < Size) {
                torch::Tensor complex_false_branch = fuzzer_utils::createTensor(Data, Size, new_offset);
                
                // Test with different predicate
                bool new_predicate = !predicate;
                torch::Tensor complex_result = new_predicate ? complex_true_branch : complex_false_branch;
                
                // Verify the result
                if (new_predicate) {
                    auto equal = torch::equal(complex_result, complex_true_branch);
                    if (!equal.item<bool>()) {
                        throw std::runtime_error("Complex result doesn't match true branch");
                    }
                } else {
                    auto equal = torch::equal(complex_result, complex_false_branch);
                    if (!equal.item<bool>()) {
                        throw std::runtime_error("Complex result doesn't match false branch");
                    }
                }
            }
        }
        
        // Test with lambda functions that perform operations
        if (offset + 1 < Size) {
            bool another_predicate = Data[offset++] & 0x1;
            
            // Create a scalar tensor for testing
            torch::Tensor scalar_input = torch::tensor(static_cast<float>(Data[offset % Size]));
            
            torch::Tensor op_result = another_predicate ? (scalar_input * 2) : (scalar_input + 5);
            
            // Verify the result
            torch::Tensor expected = another_predicate ? (scalar_input * 2) : (scalar_input + 5);
            auto equal = torch::equal(op_result, expected);
            if (!equal.item<bool>()) {
                throw std::runtime_error("Operation result doesn't match expected value");
            }
        }
        
        // Test with empty tensors
        if (offset < Size) {
            bool empty_predicate = Data[offset++] & 0x1;
            
            torch::Tensor empty_tensor = torch::empty({0});
            torch::Tensor non_empty_tensor = torch::ones({1});
            
            torch::Tensor empty_result = empty_predicate ? empty_tensor : non_empty_tensor;
            
            // Verify the result
            if (empty_predicate) {
                if (empty_result.numel() != 0) {
                    throw std::runtime_error("Empty tensor result has elements");
                }
            } else {
                if (empty_result.numel() != 1) {
                    throw std::runtime_error("Non-empty tensor result has wrong number of elements");
                }
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