#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Need at least some bytes for parameters
        if (Size < 12) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract parameters for LPPool3d from the data
        // Extract norm_type (typically 1 or 2, but allow other positive values)
        int64_t norm_type_raw;
        std::memcpy(&norm_type_raw, Data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
        double norm_type = std::abs(static_cast<double>(norm_type_raw % 10)) + 0.1;
        
        // Extract kernel sizes (3 values for D, H, W)
        int64_t k1 = (Data[offset++] % 4) + 1; // Kernel size between 1 and 4
        int64_t k2 = (Data[offset++] % 4) + 1;
        int64_t k3 = (Data[offset++] % 4) + 1;
        
        // Extract strides
        int64_t s1 = (offset < Size) ? (Data[offset++] % 4) + 1 : k1;
        int64_t s2 = (offset < Size) ? (Data[offset++] % 4) + 1 : k2;
        int64_t s3 = (offset < Size) ? (Data[offset++] % 4) + 1 : k3;
        
        // Extract ceil_mode
        bool ceil_mode = (offset < Size) ? (Data[offset++] % 2) == 1 : false;
        
        // Create a proper 5D input tensor for LPPool3d: (N, C, D, H, W)
        // Extract dimensions from data, ensuring they're large enough for the kernel
        int64_t batch_size = (offset < Size) ? (Data[offset++] % 3) + 1 : 1;
        int64_t channels = (offset < Size) ? (Data[offset++] % 4) + 1 : 1;
        int64_t depth = (offset < Size) ? (Data[offset++] % 8) + k1 : k1 + 2;
        int64_t height = (offset < Size) ? (Data[offset++] % 8) + k2 : k2 + 2;
        int64_t width = (offset < Size) ? (Data[offset++] % 8) + k3 : k3 + 2;
        
        // Ensure dimensions are valid (at least kernel size)
        depth = std::max(depth, k1);
        height = std::max(height, k2);
        width = std::max(width, k3);
        
        // Create input tensor with random data
        torch::Tensor input = torch::randn({batch_size, channels, depth, height, width}, 
                                           torch::dtype(torch::kFloat32));
        
        // Test 1: LPPool3d with single kernel size
        try {
            int64_t single_k = std::min({k1, k2, k3});
            torch::nn::LPPool3dOptions options1(norm_type, single_k);
            options1.stride(single_k);
            options1.ceil_mode(ceil_mode);
            torch::nn::LPPool3d pool1(options1);
            torch::Tensor output1 = pool1->forward(input);
        } catch (const std::exception &) {
            // Expected for some parameter combinations
        }
        
        // Test 2: LPPool3d with tuple kernel size
        try {
            torch::nn::LPPool3dOptions options2(norm_type, {k1, k2, k3});
            options2.stride({s1, s2, s3});
            options2.ceil_mode(ceil_mode);
            torch::nn::LPPool3d pool2(options2);
            torch::Tensor output2 = pool2->forward(input);
        } catch (const std::exception &) {
            // Expected for some parameter combinations
        }
        
        // Test 3: LPPool3d with ceil_mode toggled
        try {
            torch::nn::LPPool3dOptions options3(norm_type, {k1, k2, k3});
            options3.stride({s1, s2, s3});
            options3.ceil_mode(!ceil_mode);
            torch::nn::LPPool3d pool3(options3);
            torch::Tensor output3 = pool3->forward(input);
        } catch (const std::exception &) {
            // Expected for some parameter combinations
        }
        
        // Test 4: Different norm_type values
        try {
            double norm_type2 = (norm_type_raw % 2 == 0) ? 1.0 : 2.0;
            torch::nn::LPPool3dOptions options4(norm_type2, {k1, k2, k3});
            options4.stride({s1, s2, s3});
            torch::nn::LPPool3d pool4(options4);
            torch::Tensor output4 = pool4->forward(input);
        } catch (const std::exception &) {
            // Expected for some parameter combinations
        }
        
        // Test 5: With 4D input tensor (no batch dimension)
        try {
            torch::Tensor input_4d = torch::randn({channels, depth, height, width}, 
                                                   torch::dtype(torch::kFloat32));
            torch::nn::LPPool3dOptions options5(norm_type, {k1, k2, k3});
            options5.stride({s1, s2, s3});
            torch::nn::LPPool3d pool5(options5);
            torch::Tensor output5 = pool5->forward(input_4d);
        } catch (const std::exception &) {
            // Expected for some parameter combinations
        }
        
        // Test 6: With double precision tensor
        try {
            torch::Tensor input_double = torch::randn({batch_size, channels, depth, height, width}, 
                                                       torch::dtype(torch::kFloat64));
            torch::nn::LPPool3dOptions options6(norm_type, {k1, k2, k3});
            options6.stride({s1, s2, s3});
            torch::nn::LPPool3d pool6(options6);
            torch::Tensor output6 = pool6->forward(input_double);
        } catch (const std::exception &) {
            // Expected for some parameter combinations
        }
        
        // Test 7: Edge case with stride equal to 1
        try {
            torch::nn::LPPool3dOptions options7(norm_type, {k1, k2, k3});
            options7.stride(1);
            options7.ceil_mode(ceil_mode);
            torch::nn::LPPool3d pool7(options7);
            torch::Tensor output7 = pool7->forward(input);
        } catch (const std::exception &) {
            // Expected for some parameter combinations
        }
        
        // Test 8: Large norm_type
        try {
            double large_norm = static_cast<double>((norm_type_raw % 100) + 1);
            torch::nn::LPPool3dOptions options8(large_norm, {k1, k2, k3});
            options8.stride({s1, s2, s3});
            torch::nn::LPPool3d pool8(options8);
            torch::Tensor output8 = pool8->forward(input);
        } catch (const std::exception &) {
            // Expected for some parameter combinations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}