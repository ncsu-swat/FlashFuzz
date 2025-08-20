#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract min and max values for clipping if we have enough data
        double min_val = -10.0;
        double max_val = 10.0;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&min_val, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&max_val, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Test different variants of torch::clip
        
        // Variant 1: clip with both min and max
        torch::Tensor result1 = torch::clip(input, min_val, max_val);
        
        // Variant 2: clip with only min (max = None)
        torch::Tensor result2 = torch::clip(input, min_val, std::numeric_limits<double>::infinity());
        
        // Variant 3: clip with only max (min = None)
        torch::Tensor result3 = torch::clip(input, -std::numeric_limits<double>::infinity(), max_val);
        
        // Variant 4: in-place clipping
        torch::Tensor input_copy = input.clone();
        torch::Tensor result4 = torch::clip_(input_copy, min_val, max_val);
        
        // Variant 5: clip with tensor min/max values if we have enough data
        if (offset < Size) {
            try {
                torch::Tensor min_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                torch::Tensor max_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Ensure min_tensor and max_tensor are broadcastable to input
                if (min_tensor.dim() <= input.dim() && max_tensor.dim() <= input.dim()) {
                    torch::Tensor result5 = torch::clip(input, min_tensor, max_tensor);
                }
            } catch (const std::exception &) {
                // Ignore exceptions from creating additional tensors
            }
        }
        
        // Variant 6: clip with scalar Tensor min/max
        torch::Tensor min_scalar = torch::tensor(min_val);
        torch::Tensor max_scalar = torch::tensor(max_val);
        torch::Tensor result6 = torch::clip(input, min_scalar, max_scalar);
        
        // Variant 7: clip with swapped min/max (should handle this gracefully)
        if (min_val > max_val) {
            torch::Tensor result7 = torch::clip(input, max_val, min_val);
        }
        
        // Variant 8: clip with same min/max
        if (offset + sizeof(double) <= Size) {
            double same_val;
            std::memcpy(&same_val, Data + offset, sizeof(double));
            torch::Tensor result8 = torch::clip(input, same_val, same_val);
        }
        
        // Variant 9: clip with extreme values
        torch::Tensor result9 = torch::clip(input, 
                                           -std::numeric_limits<double>::max(),
                                           std::numeric_limits<double>::max());
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}