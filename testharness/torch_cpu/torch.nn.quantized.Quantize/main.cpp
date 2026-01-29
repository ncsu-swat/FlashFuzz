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
        
        // Need at least a few bytes for tensor creation and parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Quantization requires float tensors
        if (!input_tensor.is_floating_point()) {
            input_tensor = input_tensor.to(torch::kFloat32);
        }
        
        // Ensure tensor is contiguous (required for quantization)
        input_tensor = input_tensor.contiguous();
        
        // Parse scale from the remaining data
        double scale = 1.0;
        if (offset + sizeof(float) <= Size) {
            float scale_f;
            std::memcpy(&scale_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure scale is positive and reasonable
            scale = static_cast<double>(std::abs(scale_f));
            if (!std::isfinite(scale) || scale < 1e-10) scale = 1e-10;
            if (scale > 1e6) scale = 1e6;
        }
        
        // Parse zero_point from the remaining data
        int64_t zero_point = 0;
        if (offset + sizeof(int8_t) <= Size) {
            int8_t zp_byte;
            std::memcpy(&zp_byte, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            zero_point = static_cast<int64_t>(zp_byte);
        }
        
        // Parse dtype for quantization
        torch::ScalarType dtype = torch::kQInt8;
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            switch (dtype_selector % 3) {
                case 0:
                    dtype = torch::kQInt8;
                    // zero_point for qint8 should be in [-128, 127]
                    zero_point = std::max<int64_t>(-128, std::min<int64_t>(127, zero_point));
                    break;
                case 1:
                    dtype = torch::kQUInt8;
                    // zero_point for quint8 should be in [0, 255]
                    zero_point = std::max<int64_t>(0, std::min<int64_t>(255, zero_point + 128));
                    break;
                case 2:
                    dtype = torch::kQInt32;
                    // qint32 has wider range, zero_point can stay as-is
                    break;
            }
        }
        
        // Apply quantization using torch::quantize_per_tensor
        torch::Tensor quantized = torch::quantize_per_tensor(input_tensor, scale, zero_point, dtype);
        
        // Also test dequantization to exercise more code paths
        torch::Tensor dequantized = quantized.dequantize();
        
        // Test quantized tensor properties
        volatile double q_scale = quantized.q_scale();
        volatile int64_t q_zero_point = quantized.q_zero_point();
        
        // Ensure the outputs are not optimized away
        volatile bool output_exists = quantized.defined() && dequantized.defined();
        (void)output_exists;
        (void)q_scale;
        (void)q_zero_point;
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}