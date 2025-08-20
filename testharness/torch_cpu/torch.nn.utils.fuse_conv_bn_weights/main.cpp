#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create conv weights tensor
        torch::Tensor conv_w = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create conv bias tensor (can be nullptr)
        bool use_bias = offset < Size && (Data[offset++] % 2 == 0);
        torch::Tensor conv_b;
        if (use_bias && offset < Size) {
            conv_b = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Create batch norm parameters
        torch::Tensor bn_rm = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor bn_rv = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor bn_w = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor bn_b = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create eps value
        double eps = 1e-5;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure eps is positive
            eps = std::abs(eps);
            if (eps == 0.0) {
                eps = 1e-5;
            }
        }
        
        // Call fuse_conv_bn_weights
        torch::Tensor fused_weight, fused_bias;
        
        if (use_bias) {
            std::tie(fused_weight, fused_bias) = torch::fuse_conv_bn_weights(
                conv_w, conv_b, bn_rm, bn_rv, bn_w, bn_b, eps);
        } else {
            std::tie(fused_weight, fused_bias) = torch::fuse_conv_bn_weights(
                conv_w, torch::Tensor(), bn_rm, bn_rv, bn_w, bn_b, eps);
        }
        
        // Verify the output tensors are valid
        if (fused_weight.defined()) {
            auto sizes = fused_weight.sizes();
            auto numel = fused_weight.numel();
        }
        
        if (fused_bias.defined()) {
            auto sizes = fused_bias.sizes();
            auto numel = fused_bias.numel();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}