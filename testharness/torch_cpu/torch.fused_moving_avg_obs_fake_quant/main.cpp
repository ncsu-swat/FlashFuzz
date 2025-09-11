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
        
        // Early exit if not enough data
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor X = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for fused_moving_avg_obs_fake_quant
        // Ensure we have enough data for parameters
        if (offset + 8 > Size) {
            return 0;
        }
        
        // Extract scale
        float scale = 0.1f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Extract zero_point
        int64_t zero_point = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Extract averaging constant
        double averaging_const = 0.01;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&averaging_const, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure averaging_const is between 0 and 1
            averaging_const = std::abs(averaging_const);
            if (averaging_const > 1.0) {
                averaging_const = averaging_const - std::floor(averaging_const);
            }
        }
        
        // Extract quant_min and quant_max
        int64_t quant_min = 0;
        int64_t quant_max = 255;
        if (offset + 2 * sizeof(int64_t) <= Size) {
            std::memcpy(&quant_min, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            std::memcpy(&quant_max, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Extract ch_axis
        int64_t ch_axis = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&ch_axis, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Ensure ch_axis is within tensor dimension range
            if (X.dim() > 0) {
                ch_axis = ch_axis % X.dim();
                if (ch_axis < 0) {
                    ch_axis += X.dim();
                }
            } else {
                ch_axis = 0;
            }
        }
        
        // Create observer state tensors
        torch::Tensor running_min;
        torch::Tensor running_max;
        torch::Tensor scale_tensor;
        torch::Tensor zero_point_tensor;
        
        // Initialize observer state tensors
        if (X.dim() > 0 && X.size(ch_axis) > 0) {
            running_min = torch::full({X.size(ch_axis)}, std::numeric_limits<float>::max(), torch::kFloat);
            running_max = torch::full({X.size(ch_axis)}, std::numeric_limits<float>::lowest(), torch::kFloat);
            scale_tensor = torch::full({X.size(ch_axis)}, scale, torch::kFloat);
            zero_point_tensor = torch::full({X.size(ch_axis)}, zero_point, torch::kInt);
        } else {
            running_min = torch::tensor(std::numeric_limits<float>::max(), torch::kFloat);
            running_max = torch::tensor(std::numeric_limits<float>::lowest(), torch::kFloat);
            scale_tensor = torch::tensor(scale, torch::kFloat);
            zero_point_tensor = torch::tensor(zero_point, torch::kInt);
        }
        
        // Create observer_on and fake_quant_on flags as tensors
        torch::Tensor observer_on = torch::tensor((offset < Size) ? (Data[offset++] % 2 == 0) : true, torch::kBool);
        torch::Tensor fake_quant_on = torch::tensor((offset < Size) ? (Data[offset++] % 2 == 0) : true, torch::kBool);
        
        // Extract per_row_fake_quant and symmetric_quant flags
        bool per_row_fake_quant = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
        bool symmetric_quant = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
        
        // Call fused_moving_avg_obs_fake_quant
        torch::Tensor output = torch::fused_moving_avg_obs_fake_quant(
            X, 
            observer_on,
            fake_quant_on,
            running_min, 
            running_max, 
            scale_tensor, 
            zero_point_tensor, 
            averaging_const, 
            quant_min, 
            quant_max, 
            ch_axis, 
            per_row_fake_quant, 
            symmetric_quant
        );
        
        // Ensure the output is used to prevent optimization
        if (output.defined()) {
            volatile float sum = output.sum().item<float>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
