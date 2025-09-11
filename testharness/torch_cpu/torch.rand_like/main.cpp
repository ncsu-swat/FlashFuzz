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
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply rand_like operation with various options
        if (offset + 1 < Size) {
            // Use the next byte to determine options
            uint8_t options_byte = Data[offset++];
            
            // Extract option flags from the byte
            bool requires_grad = (options_byte & 0x01);
            bool pin_memory = (options_byte & 0x02);
            
            // Get a data type for the output tensor
            torch::ScalarType output_dtype = fuzzer_utils::parseDataType(
                (offset < Size) ? Data[offset++] : 0);
            
            // Create tensor options
            auto tensor_options = torch::TensorOptions()
                .dtype(output_dtype)
                .requires_grad(requires_grad)
                .pinned_memory(pin_memory);
            
            // Apply rand_like with different options
            torch::Tensor result;
            
            // Test different variants of rand_like
            if (offset < Size) {
                uint8_t variant = Data[offset++] % 3;
                
                switch (variant) {
                    case 0:
                        // Basic rand_like
                        result = torch::rand_like(input_tensor);
                        break;
                    case 1:
                        // rand_like with options
                        result = torch::rand_like(input_tensor, tensor_options);
                        break;
                    case 2:
                        // rand_like with memory format
                        if (offset < Size) {
                            uint8_t format_byte = Data[offset++] % 4;
                            c10::MemoryFormat memory_format;
                            
                            switch (format_byte) {
                                case 0:
                                    memory_format = c10::MemoryFormat::Contiguous;
                                    break;
                                case 1:
                                    memory_format = c10::MemoryFormat::ChannelsLast;
                                    break;
                                case 2:
                                    memory_format = c10::MemoryFormat::ChannelsLast3d;
                                    break;
                                default:
                                    memory_format = c10::MemoryFormat::Preserve;
                                    break;
                            }
                            
                            result = torch::rand_like(input_tensor, tensor_options.memory_format(memory_format));
                        } else {
                            result = torch::rand_like(input_tensor, tensor_options);
                        }
                        break;
                }
            } else {
                // Default to basic rand_like if we don't have enough data
                result = torch::rand_like(input_tensor);
            }
            
            // Basic validation - check that result has same shape as input
            if (result.sizes() != input_tensor.sizes()) {
                throw std::runtime_error("rand_like produced tensor with different shape");
            }
            
            // Check that values are in the expected range [0, 1)
            auto min_val = torch::min(result).item<float>();
            auto max_val = torch::max(result).item<float>();
            
            if (min_val < 0.0f || max_val >= 1.0f) {
                throw std::runtime_error("rand_like produced values outside [0, 1) range");
            }
        } else {
            // Basic case with default options
            torch::Tensor result = torch::rand_like(input_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
