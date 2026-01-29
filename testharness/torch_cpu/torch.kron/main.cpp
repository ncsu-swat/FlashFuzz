#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create first input tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second input tensor if we have enough data left
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If not enough data for second tensor, create a simple one
            tensor2 = torch::ones({1, 1});
        }
        
        // Apply kron operation - this is the main API under test
        torch::Tensor result = torch::kron(tensor1, tensor2);
        
        // Optional: Try the operation with tensors swapped
        try {
            if (offset + 1 < Size && Data[offset] % 2 == 0) {
                torch::Tensor result2 = torch::kron(tensor2, tensor1);
            }
        } catch (...) {
            // Silently ignore expected failures
        }
        
        // Optional: Try with scalar tensors
        try {
            if (offset + 1 < Size && Data[offset] % 3 == 0) {
                torch::Tensor scalar1 = torch::tensor(3.14);
                torch::Tensor scalar2 = torch::tensor(2.71);
                torch::Tensor result3 = torch::kron(scalar1, tensor1);
                torch::Tensor result4 = torch::kron(tensor1, scalar2);
                torch::Tensor result5 = torch::kron(scalar1, scalar2);
            }
        } catch (...) {
            // Silently ignore expected failures
        }
        
        // Optional: Try with empty tensors
        try {
            if (offset + 1 < Size && Data[offset] % 5 == 0) {
                torch::Tensor empty1 = torch::empty({0, 2});
                torch::Tensor empty2 = torch::empty({2, 0});
                torch::Tensor result6 = torch::kron(empty1, tensor1);
                torch::Tensor result7 = torch::kron(tensor1, empty2);
                torch::Tensor result8 = torch::kron(empty1, empty2);
            }
        } catch (...) {
            // Silently ignore expected failures
        }
        
        // Optional: Try with boolean tensors
        try {
            if (offset + 1 < Size && Data[offset] % 7 == 0) {
                torch::Tensor bool1 = torch::tensor({{true, false}, {false, true}});
                torch::Tensor result9 = torch::kron(bool1, tensor1);
                torch::Tensor result10 = torch::kron(tensor1, bool1);
            }
        } catch (...) {
            // Silently ignore expected failures
        }
        
        // Additional coverage: Try with different dtypes
        try {
            if (offset + 1 < Size && Data[offset] % 11 == 0) {
                torch::Tensor int_tensor = torch::randint(0, 10, {2, 2}, torch::kInt32);
                torch::Tensor result11 = torch::kron(int_tensor, tensor1);
            }
        } catch (...) {
            // Silently ignore expected failures
        }
        
        // Additional coverage: Try with complex tensors
        try {
            if (offset + 1 < Size && Data[offset] % 13 == 0) {
                torch::Tensor complex_tensor = torch::randn({2, 2}, torch::kComplexFloat);
                torch::Tensor result12 = torch::kron(complex_tensor, complex_tensor);
            }
        } catch (...) {
            // Silently ignore expected failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}