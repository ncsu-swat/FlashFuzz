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
        
        // Need at least 2 bytes for basic parameters
        if (Size < 2) {
            return 0;
        }
        
        // Parse number of dimensions
        uint8_t rank_byte = Data[offset++];
        uint8_t rank = fuzzer_utils::parseRank(rank_byte);
        
        // Parse shape for randn
        std::vector<int64_t> shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
        
        // Parse dtype for the tensor
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Create options with the selected dtype
            auto options = torch::TensorOptions().dtype(dtype);
            
            // Call torch::randn with the shape and options
            torch::Tensor result = torch::randn(shape, options);
            
            // Test some basic properties to ensure the tensor was created correctly
            if (result.dim() != rank) {
                throw std::runtime_error("Dimension mismatch");
            }
            
            // Test additional operations on the result tensor
            if (Size > offset && offset < Size - 1) {
                uint8_t op_selector = Data[offset++];
                
                // Perform different operations based on the selector
                switch (op_selector % 5) {
                    case 0:
                        // Test mean and std
                        result.mean();
                        result.std();
                        break;
                    case 1:
                        // Test mathematical operations
                        result = result * 2.0;
                        result = result + 1.0;
                        break;
                    case 2:
                        // Test reshaping if possible
                        if (!result.numel()) break;
                        try {
                            result = result.reshape({-1});
                        } catch (...) {
                            // Reshape might fail, that's okay
                        }
                        break;
                    case 3:
                        {
                            // Test cloning and other operations
                            torch::Tensor cloned = result.clone();
                            cloned = torch::abs(cloned);
                            break;
                        }
                    case 4:
                        // Test type conversion
                        try {
                            result = result.to(torch::kFloat);
                        } catch (...) {
                            // Conversion might fail, that's okay
                        }
                        break;
                }
            }
            
            // Test randn_like
            if (Size > offset) {
                torch::Tensor randn_like_result = torch::randn_like(result);
            }
        } else {
            // If no dtype specified, use default float
            torch::Tensor result = torch::randn(shape);
        }
        
        // Test randn with specific mean and std if we have more data
        if (Size > offset + 1) {
            // Extract values for mean and std
            float mean_val = static_cast<float>(Data[offset++]) / 255.0f * 10.0f - 5.0f; // Range: -5 to 5
            float std_val = static_cast<float>(Data[offset++]) / 255.0f * 5.0f + 0.1f;   // Range: 0.1 to 5.1
            
            // Create a tensor with the specified mean and std
            torch::Tensor custom_randn = torch::randn(shape) * std_val + mean_val;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
