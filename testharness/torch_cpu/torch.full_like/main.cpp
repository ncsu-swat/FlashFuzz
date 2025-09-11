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
        
        // Need at least a few bytes for the input tensor and fill value
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract fill value from the remaining data
        double fill_value = 0.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&fill_value, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Create a scalar tensor for the fill value
        torch::Scalar fill_scalar(fill_value);
        
        // Try different options for full_like
        try {
            // Basic full_like with default options
            torch::Tensor result1 = torch::full_like(input_tensor, fill_scalar);
            
            // Check if we have more data to use for options
            if (offset < Size) {
                // Extract dtype selector
                uint8_t dtype_selector = Data[offset++];
                torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
                
                // full_like with specified dtype
                torch::Tensor result2 = torch::full_like(
                    input_tensor, 
                    fill_scalar, 
                    torch::TensorOptions().dtype(dtype)
                );
                
                // Try with layout, device, requires_grad options if we have more data
                if (offset < Size) {
                    bool requires_grad = (Data[offset++] % 2) == 1;
                    
                    torch::Tensor result3 = torch::full_like(
                        input_tensor,
                        fill_scalar,
                        torch::TensorOptions()
                            .dtype(dtype)
                            .layout(torch::kStrided)
                            .requires_grad(requires_grad)
                    );
                    
                    // Try with memory_format if we have more data
                    if (offset < Size) {
                        uint8_t memory_format_selector = Data[offset++];
                        torch::MemoryFormat memory_format;
                        
                        // Select a memory format based on the input data
                        switch (memory_format_selector % 4) {
                            case 0:
                                memory_format = torch::MemoryFormat::Contiguous;
                                break;
                            case 1:
                                memory_format = torch::MemoryFormat::ChannelsLast;
                                break;
                            case 2:
                                memory_format = torch::MemoryFormat::ChannelsLast3d;
                                break;
                            default:
                                memory_format = torch::MemoryFormat::Preserve;
                                break;
                        }
                        
                        torch::Tensor result4 = torch::full_like(
                            input_tensor,
                            fill_scalar,
                            torch::TensorOptions()
                                .dtype(dtype)
                                .layout(torch::kStrided)
                                .requires_grad(requires_grad),
                            memory_format
                        );
                    }
                }
            }
            
            // Try with different fill values if we have more data
            if (offset + sizeof(int64_t) <= Size) {
                int64_t int_fill_value;
                std::memcpy(&int_fill_value, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                torch::Scalar int_scalar(int_fill_value);
                torch::Tensor result5 = torch::full_like(input_tensor, int_scalar);
            }
            
            // Try with boolean fill value
            if (offset < Size) {
                bool bool_fill_value = (Data[offset++] % 2) == 1;
                torch::Scalar bool_scalar(bool_fill_value);
                torch::Tensor result6 = torch::full_like(input_tensor, bool_scalar);
            }
        }
        catch (const c10::Error &e) {
            // Catch PyTorch-specific errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
