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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Try different options for as_tensor
        try {
            // Basic as_tensor call - use from_blob to simulate as_tensor behavior
            torch::Tensor result1 = input_tensor.clone();
            
            // Try with different dtype
            if (offset + 1 < Size) {
                auto dtype = fuzzer_utils::parseDataType(Data[offset++]);
                torch::Tensor result2 = input_tensor.to(dtype);
            }
            
            // Try with different device
            if (offset + 1 < Size) {
                bool use_cuda = Data[offset++] % 2 == 0 && torch::cuda::is_available();
                auto device = use_cuda ? torch::kCUDA : torch::kCPU;
                torch::Tensor result3 = input_tensor.to(device);
            }
            
            // Try with both dtype and device
            if (offset + 2 < Size) {
                auto dtype = fuzzer_utils::parseDataType(Data[offset++]);
                bool use_cuda = Data[offset++] % 2 == 0 && torch::cuda::is_available();
                auto device = use_cuda ? torch::kCUDA : torch::kCPU;
                torch::Tensor result4 = input_tensor.to(torch::TensorOptions().dtype(dtype).device(device));
            }
            
            // Try with non-tensor inputs if we have enough data
            if (offset + 4 < Size) {
                // Create a vector from the remaining data
                std::vector<int> vec_data;
                size_t remaining = std::min(Size - offset, static_cast<size_t>(16));
                for (size_t i = 0; i < remaining; i++) {
                    vec_data.push_back(static_cast<int>(Data[offset++]));
                }
                
                // Try tensor creation with vector
                torch::Tensor result5 = torch::tensor(vec_data);
                
                // Try with dtype
                if (offset < Size) {
                    auto dtype = fuzzer_utils::parseDataType(Data[offset++]);
                    torch::Tensor result6 = torch::tensor(vec_data, torch::TensorOptions().dtype(dtype));
                }
            }
            
            // Try with scalar inputs
            if (offset < Size) {
                int scalar_val = static_cast<int>(Data[offset++]);
                torch::Tensor result7 = torch::tensor(scalar_val);
                
                if (offset < Size) {
                    auto dtype = fuzzer_utils::parseDataType(Data[offset++]);
                    torch::Tensor result8 = torch::tensor(scalar_val, torch::TensorOptions().dtype(dtype));
                }
            }
            
            // Try with empty inputs
            std::vector<int> empty_vec;
            torch::Tensor result9 = torch::tensor(empty_vec);
            
            // Try with nested vectors
            if (offset + 4 < Size) {
                std::vector<std::vector<int>> nested_vec;
                size_t num_inner = Data[offset++] % 3 + 1;
                size_t inner_size = Data[offset++] % 3 + 1;
                
                for (size_t i = 0; i < num_inner; i++) {
                    std::vector<int> inner;
                    for (size_t j = 0; j < inner_size && offset < Size; j++) {
                        inner.push_back(static_cast<int>(Data[offset++]));
                    }
                    nested_vec.push_back(inner);
                }
                
                torch::Tensor result10 = torch::tensor(nested_vec);
            }
        }
        catch (const c10::Error &e) {
            // PyTorch specific errors are expected and okay
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
