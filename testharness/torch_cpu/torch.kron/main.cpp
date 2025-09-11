#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
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
        
        // Apply kron operation
        torch::Tensor result = torch::kron(tensor1, tensor2);
        
        // Optional: Try the operation with tensors swapped
        if (offset + 1 < Size && Data[offset] % 2 == 0) {
            torch::Tensor result2 = torch::kron(tensor2, tensor1);
        }
        
        // Optional: Try with scalar tensors
        if (offset + 1 < Size && Data[offset] % 3 == 0) {
            torch::Tensor scalar1 = torch::tensor(3.14);
            torch::Tensor scalar2 = torch::tensor(2.71);
            torch::Tensor result3 = torch::kron(scalar1, tensor1);
            torch::Tensor result4 = torch::kron(tensor1, scalar2);
            torch::Tensor result5 = torch::kron(scalar1, scalar2);
        }
        
        // Optional: Try with empty tensors
        if (offset + 1 < Size && Data[offset] % 5 == 0) {
            torch::Tensor empty1 = torch::empty({0, 2});
            torch::Tensor empty2 = torch::empty({2, 0});
            torch::Tensor result6 = torch::kron(empty1, tensor1);
            torch::Tensor result7 = torch::kron(tensor1, empty2);
            torch::Tensor result8 = torch::kron(empty1, empty2);
        }
        
        // Optional: Try with boolean tensors
        if (offset + 1 < Size && Data[offset] % 7 == 0) {
            torch::Tensor bool1 = torch::tensor({{true, false}, {false, true}});
            torch::Tensor result9 = torch::kron(bool1, tensor1);
            torch::Tensor result10 = torch::kron(tensor1, bool1);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
