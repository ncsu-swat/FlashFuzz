#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic parameters for the operation
        if (Size < 32) {
            return 0;
        }

        // Extract basic parameters
        auto batch_size = extract_int(Data, Size, offset, 1, 128);
        auto channels = extract_int(Data, Size, offset, 1, 64);
        auto height = extract_int(Data, Size, offset, 1, 32);
        auto width = extract_int(Data, Size, offset, 1, 32);
        
        // Extract momentum parameter (0.0 to 1.0)
        auto momentum = extract_float(Data, Size, offset, 0.0f, 1.0f);
        
        // Extract exponential_average_factor parameter
        auto exp_avg_factor = extract_float(Data, Size, offset, 0.0f, 1.0f);
        
        // Extract device and dtype parameters
        auto device_type = extract_int(Data, Size, offset, 0, 1); // 0=CPU, 1=CUDA if available
        auto dtype_idx = extract_int(Data, Size, offset, 0, 2); // 0=float32, 1=float64, 2=float16
        
        // Map dtype
        torch::ScalarType dtype;
        switch (dtype_idx) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kFloat16; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Determine device
        torch::Device device = torch::kCPU;
        if (device_type == 1 && torch::cuda::is_available()) {
            device = torch::kCUDA;
        }
        
        // Create input tensor with NCHW format for batch norm
        auto input = torch::randn({batch_size, channels, height, width}, 
                                 torch::TensorOptions().dtype(dtype).device(device));
        
        // Create running mean and var tensors (1D with size = channels)
        auto running_mean = torch::randn({channels}, 
                                        torch::TensorOptions().dtype(dtype).device(device));
        auto running_var = torch::abs(torch::randn({channels}, 
                                     torch::TensorOptions().dtype(dtype).device(device))) + 1e-5f;
        
        // Test different scenarios based on remaining data
        auto test_scenario = extract_int(Data, Size, offset, 0, 7);
        
        switch (test_scenario) {
            case 0: {
                // Basic case with normal parameters
                auto result = torch::batch_norm_update_stats(input, running_mean, running_var, momentum);
                break;
            }
            case 1: {
                // Test with zero momentum
                auto result = torch::batch_norm_update_stats(input, running_mean, running_var, 0.0);
                break;
            }
            case 2: {
                // Test with momentum = 1.0
                auto result = torch::batch_norm_update_stats(input, running_mean, running_var, 1.0);
                break;
            }
            case 3: {
                // Test with very small input
                auto small_input = torch::randn({1, 1, 1, 1}, 
                                              torch::TensorOptions().dtype(dtype).device(device));
                auto small_mean = torch::randn({1}, 
                                             torch::TensorOptions().dtype(dtype).device(device));
                auto small_var = torch::abs(torch::randn({1}, 
                                           torch::TensorOptions().dtype(dtype).device(device))) + 1e-5f;
                auto result = torch::batch_norm_update_stats(small_input, small_mean, small_var, momentum);
                break;
            }
            case 4: {
                // Test with extreme values in input
                auto extreme_input = input * 1000.0f;
                auto result = torch::batch_norm_update_stats(extreme_input, running_mean, running_var, momentum);
                break;
            }
            case 5: {
                // Test with very small values in input
                auto tiny_input = input * 1e-6f;
                auto result = torch::batch_norm_update_stats(tiny_input, running_mean, running_var, momentum);
                break;
            }
            case 6: {
                // Test with different tensor shapes - 2D case (NC format)
                auto input_2d = torch::randn({batch_size, channels}, 
                                           torch::TensorOptions().dtype(dtype).device(device));
                auto result = torch::batch_norm_update_stats(input_2d, running_mean, running_var, momentum);
                break;
            }
            case 7: {
                // Test with 3D input (NCL format)
                auto length = extract_int(Data, Size, offset, 1, 32);
                auto input_3d = torch::randn({batch_size, channels, length}, 
                                           torch::TensorOptions().dtype(dtype).device(device));
                auto result = torch::batch_norm_update_stats(input_3d, running_mean, running_var, momentum);
                break;
            }
        }
        
        // Test edge cases with remaining data if available
        if (offset < Size - 4) {
            auto edge_case = extract_int(Data, Size, offset, 0, 3);
            
            switch (edge_case) {
                case 0: {
                    // Test with NaN values in running stats
                    auto nan_mean = running_mean.clone();
                    nan_mean[0] = std::numeric_limits<float>::quiet_NaN();
                    try {
                        auto result = torch::batch_norm_update_stats(input, nan_mean, running_var, momentum);
                    } catch (...) {
                        // Expected to potentially fail
                    }
                    break;
                }
                case 1: {
                    // Test with infinity values
                    auto inf_var = running_var.clone();
                    inf_var[0] = std::numeric_limits<float>::infinity();
                    try {
                        auto result = torch::batch_norm_update_stats(input, running_mean, inf_var, momentum);
                    } catch (...) {
                        // Expected to potentially fail
                    }
                    break;
                }
                case 2: {
                    // Test with zero variance
                    auto zero_var = torch::zeros_like(running_var);
                    try {
                        auto result = torch::batch_norm_update_stats(input, running_mean, zero_var, momentum);
                    } catch (...) {
                        // Expected to potentially fail
                    }
                    break;
                }
                case 3: {
                    // Test with negative variance
                    auto neg_var = -torch::abs(running_var);
                    try {
                        auto result = torch::batch_norm_update_stats(input, running_mean, neg_var, momentum);
                    } catch (...) {
                        // Expected to potentially fail
                    }
                    break;
                }
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}