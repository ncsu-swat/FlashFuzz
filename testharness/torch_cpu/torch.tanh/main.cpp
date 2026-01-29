#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor for tanh operation
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply tanh operation
        torch::Tensor output = torch::tanh(input);
        
        // Try in-place version if there's more data
        if (offset < Size) {
            uint8_t choice = Data[offset++];
            if (choice % 2 == 0) {
                torch::Tensor input_copy = input.clone();
                input_copy.tanh_();
            }
        }
        
        // Try with out parameter version if there's more data
        if (offset + 2 < Size) {
            size_t offset2 = 0;
            const uint8_t* remaining_data = Data + offset;
            size_t remaining_size = Size - offset;
            
            // Create another tensor with remaining data
            torch::Tensor input2 = fuzzer_utils::createTensor(remaining_data, remaining_size, offset2);
            offset += offset2;
            
            // Try tanh with out parameter
            torch::Tensor out = torch::empty_like(input2);
            torch::tanh_out(out, input2);
        }
        
        // Try tanh with different dtypes
        if (offset + 1 < Size) {
            uint8_t dtype_choice = Data[offset++];
            
            try {
                // Convert to float types that support tanh
                torch::Dtype target_dtype;
                switch (dtype_choice % 4) {
                    case 0: target_dtype = torch::kFloat32; break;
                    case 1: target_dtype = torch::kFloat64; break;
                    case 2: target_dtype = torch::kFloat16; break;
                    default: target_dtype = torch::kFloat32; break;
                }
                
                torch::Tensor input_converted = input.to(target_dtype);
                torch::Tensor output_converted = torch::tanh(input_converted);
            } catch (...) {
                // Silently ignore dtype conversion failures
            }
        }
        
        // Try with requires_grad for autograd coverage
        if (offset < Size && (Data[offset++] % 3 == 0)) {
            try {
                torch::Tensor grad_input = input.to(torch::kFloat32).clone().detach().requires_grad_(true);
                torch::Tensor grad_output = torch::tanh(grad_input);
                grad_output.sum().backward();
            } catch (...) {
                // Silently ignore backward errors (expected for some tensor configs)
            }
        }
        
        // Test with complex tensors if supported
        if (offset < Size && (Data[offset++] % 4 == 0)) {
            try {
                torch::Tensor complex_input = torch::complex(
                    input.to(torch::kFloat32),
                    torch::zeros_like(input).to(torch::kFloat32)
                );
                torch::Tensor complex_output = torch::tanh(complex_input);
            } catch (...) {
                // Silently ignore complex tensor errors
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}