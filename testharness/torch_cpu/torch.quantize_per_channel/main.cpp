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
        
        // Need at least some data to proceed
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create scales tensor (1D tensor of float values)
        torch::Tensor scales;
        if (offset < Size) {
            scales = fuzzer_utils::createTensor(Data, Size, offset);
            // Ensure scales is 1D and has positive values
            if (scales.dim() > 0) {
                scales = scales.abs().reshape({scales.numel()});
            } else {
                scales = torch::ones({1});
            }
        } else {
            scales = torch::ones({1});
        }
        
        // Create zero_points tensor (1D tensor of int values)
        torch::Tensor zero_points;
        if (offset < Size) {
            zero_points = fuzzer_utils::createTensor(Data, Size, offset);
            // Ensure zero_points is 1D and has integer values
            if (zero_points.dim() > 0) {
                zero_points = zero_points.reshape({zero_points.numel()}).to(torch::kInt);
            } else {
                zero_points = torch::zeros({1}, torch::kInt);
            }
        } else {
            zero_points = torch::zeros({1}, torch::kInt);
        }
        
        // Get axis parameter (must be in range [-input_tensor.dim(), input_tensor.dim()-1])
        int64_t axis = 0;
        if (offset < Size && input_tensor.dim() > 0) {
            uint8_t axis_byte = Data[offset++];
            axis = static_cast<int64_t>(axis_byte) % (2 * input_tensor.dim()) - input_tensor.dim();
        }
        
        // Get dtype
        torch::ScalarType dtype = torch::kQUInt8;
        if (offset < Size) {
            uint8_t dtype_byte = Data[offset++];
            if (dtype_byte % 2 == 0) {
                dtype = torch::kQUInt8;
            } else {
                dtype = torch::kQInt8;
            }
        }
        
        // Ensure scales and zero_points have the right size for the specified axis
        if (input_tensor.dim() > 0) {
            int64_t axis_size = input_tensor.size(axis < 0 ? axis + input_tensor.dim() : axis);
            
            if (scales.numel() != axis_size) {
                scales = scales.numel() > 0 ? 
                    scales.index({torch::indexing::Slice(0, scales.numel())}).repeat(axis_size / scales.numel() + 1).index({torch::indexing::Slice(0, axis_size)}) :
                    torch::ones({axis_size});
            }
            
            if (zero_points.numel() != axis_size) {
                zero_points = zero_points.numel() > 0 ?
                    zero_points.index({torch::indexing::Slice(0, zero_points.numel())}).repeat(axis_size / zero_points.numel() + 1).index({torch::indexing::Slice(0, axis_size)}) :
                    torch::zeros({axis_size}, torch::kInt);
            }
        }
        
        // Apply quantize_per_channel
        torch::Tensor quantized;
        
        // Convert input to float if needed
        if (input_tensor.scalar_type() != torch::kFloat) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Call quantize_per_channel
        quantized = torch::quantize_per_channel(
            input_tensor,
            scales,
            zero_points,
            axis,
            dtype
        );
        
        // Test dequantization as well
        torch::Tensor dequantized = quantized.dequantize();
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
