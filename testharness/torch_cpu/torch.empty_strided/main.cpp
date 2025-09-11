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
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Parse size information
        uint8_t rank_byte = Data[offset++];
        uint8_t rank = fuzzer_utils::parseRank(rank_byte);
        
        // Parse shape for the tensor
        std::vector<int64_t> shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
        
        // Parse data type
        uint8_t dtype_selector = 0;
        if (offset < Size) {
            dtype_selector = Data[offset++];
        }
        torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
        
        // Parse strides
        std::vector<int64_t> strides;
        for (size_t i = 0; i < shape.size() && offset + sizeof(int64_t) <= Size; i++) {
            int64_t stride_val;
            std::memcpy(&stride_val, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            strides.push_back(stride_val);
        }
        
        // If we don't have enough strides, fill with default values
        while (strides.size() < shape.size()) {
            strides.push_back(1);
        }
        
        // Create options with the specified dtype
        auto options = torch::TensorOptions().dtype(dtype);
        
        // Call empty_strided with the parsed parameters
        torch::Tensor result;
        try {
            result = torch::empty_strided(shape, strides, options);
            
            // Basic checks to ensure the tensor was created correctly
            if (result.sizes() != shape) {
                throw std::runtime_error("Created tensor has incorrect shape");
            }
            
            // Check strides
            for (size_t i = 0; i < shape.size(); i++) {
                if (result.stride(i) != strides[i]) {
                    throw std::runtime_error("Created tensor has incorrect strides");
                }
            }
            
            // Perform some operations on the tensor to ensure it's valid
            if (result.numel() > 0) {
                result.zero_();
                result.fill_(1.0);
            }
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected for invalid inputs
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
