#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for the input tensor and parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for randint_like from the remaining data
        int64_t low = 0;
        int64_t high = 10;
        
        // Parse low and high values if we have enough data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&low, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&high, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Ensure high > low (required by randint_like)
        if (high <= low) {
            high = low + 1;
        }
        
        // Parse dtype for output tensor if we have enough data
        torch::ScalarType dtype = input_tensor.scalar_type();
        if (offset < Size) {
            dtype = fuzzer_utils::parseDataType(Data[offset++]);
        }
        
        // Parse device for output tensor
        torch::Device device = torch::kCPU;
        
        // Create options for randint_like
        auto options = torch::TensorOptions().dtype(dtype).device(device);
        
        // Call randint_like with different parameter combinations
        torch::Tensor output1 = torch::randint_like(input_tensor, high);
        torch::Tensor output2 = torch::randint_like(input_tensor, low, high);
        torch::Tensor output3 = torch::randint_like(input_tensor, high, options);
        torch::Tensor output4 = torch::randint_like(input_tensor, low, high, options);
        
        // Verify basic properties
        assert(output1.sizes() == input_tensor.sizes());
        assert(output2.sizes() == input_tensor.sizes());
        assert(output3.sizes() == input_tensor.sizes());
        assert(output4.sizes() == input_tensor.sizes());
        
        assert(output3.scalar_type() == dtype);
        assert(output4.scalar_type() == dtype);
        
        // Test with empty tensor
        if (input_tensor.numel() == 0) {
            torch::Tensor empty_output = torch::randint_like(input_tensor, low, high, options);
            assert(empty_output.numel() == 0);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}