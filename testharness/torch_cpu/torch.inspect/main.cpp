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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // torch.inspect doesn't exist in PyTorch C++ API
        // Instead, we can print tensor information using std::cout
        std::cout << "Tensor: " << tensor << std::endl;
        torch::Tensor result = tensor;
        
        // Try with a custom message
        if (offset < Size) {
            // Use some bytes as a message length
            size_t msg_len = Data[offset++] % 10;
            if (offset + msg_len <= Size) {
                std::string message(reinterpret_cast<const char*>(Data + offset), msg_len);
                offset += msg_len;
                std::cout << message << ": " << tensor << std::endl;
                torch::Tensor result_with_msg = tensor;
            }
        }
        
        // Try inspecting multiple tensors
        if (Size - offset > 2) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            std::cout << "Tensor1: " << tensor << ", Tensor2: " << tensor2 << std::endl;
            
            // Try with a custom message for multiple tensors
            if (offset < Size) {
                size_t msg_len = Data[offset++] % 10;
                if (offset + msg_len <= Size) {
                    std::string message(reinterpret_cast<const char*>(Data + offset), msg_len);
                    offset += msg_len;
                    std::cout << message << " - Tensor1: " << tensor << ", Tensor2: " << tensor2 << std::endl;
                }
            }
        }
        
        // Try with empty tensor
        torch::Tensor empty_tensor = torch::empty({0});
        std::cout << "Empty tensor: " << empty_tensor << std::endl;
        
        // Try with scalar tensor
        torch::Tensor scalar_tensor = torch::tensor(3.14);
        std::cout << "Scalar tensor: " << scalar_tensor << std::endl;
        
        // Try with boolean tensor
        torch::Tensor bool_tensor = torch::tensor(true);
        std::cout << "Boolean tensor: " << bool_tensor << std::endl;
        
        // Try with complex tensor if we have enough data
        if (offset < Size) {
            torch::Tensor complex_tensor = torch::complex(
                torch::ones({2, 2}), 
                torch::ones({2, 2})
            );
            std::cout << "Complex tensor: " << complex_tensor << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
