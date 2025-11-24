#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
// Target API: torch.as_tensor

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
            
            // Try with nested vectors by flattening into a small 2D tensor
            if (offset + 2 < Size) {
                size_t num_inner = Data[offset++] % 3 + 1;
                size_t inner_size = Data[offset++] % 3 + 1;
                size_t total_elems = num_inner * inner_size;

                std::vector<int64_t> flat(total_elems, 0);
                for (size_t i = 0; i < total_elems && offset < Size; i++) {
                    flat[i] = static_cast<int64_t>(Data[offset++]);
                }

                torch::Tensor flat_tensor = torch::tensor(flat, torch::TensorOptions().dtype(torch::kLong));
                torch::Tensor result10 = flat_tensor.view({static_cast<int64_t>(num_inner), static_cast<int64_t>(inner_size)});
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
