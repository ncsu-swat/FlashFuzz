#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract padding parameters from the remaining data
        uint8_t padding_mode_byte = 0;
        if (offset < Size) {
            padding_mode_byte = Data[offset++];
        }
        
        // Select padding mode for functional API
        torch::nn::functional::PadFuncOptions::mode_t padding_mode;
        switch (padding_mode_byte % 4) {
            case 0:
                padding_mode = torch::kConstant;
                break;
            case 1:
                padding_mode = torch::kReflect;
                break;
            case 2:
                padding_mode = torch::kReplicate;
                break;
            case 3:
                padding_mode = torch::kCircular;
                break;
        }
        
        // Extract padding values (keep them reasonable: 0-15)
        std::vector<int64_t> padding;
        uint8_t padding_size = 0;
        if (offset < Size) {
            padding_size = (Data[offset++] % 6) + 1; // 1-6 padding values
        }
        
        for (uint8_t i = 0; i < padding_size && offset < Size; i++) {
            // Keep padding values small and non-negative (0-31)
            int64_t pad_value = static_cast<int64_t>(Data[offset++] % 32);
            padding.push_back(pad_value);
        }
        
        // Ensure we have at least 2 padding values
        while (padding.size() < 2) {
            padding.push_back(1);
        }
        
        // Extract a value for constant padding
        double pad_value = 0.0;
        if (offset < Size) {
            pad_value = static_cast<double>(Data[offset++]) / 255.0 * 10.0 - 5.0;
        }
        
        // 1. Test ConstantPad1d (expects 3D input: N, C, W)
        try {
            auto input_3d = input.dim() == 3 ? input : 
                           (input.dim() < 3 ? input.unsqueeze(0).unsqueeze(0) : input.flatten().unsqueeze(0).unsqueeze(0));
            // Ensure reasonable size
            if (input_3d.size(-1) > 0) {
                auto options = torch::nn::ConstantPad1dOptions({padding[0] % 16, padding[1] % 16}, pad_value);
                auto module = torch::nn::ConstantPad1d(options);
                auto output = module->forward(input_3d);
            }
        } catch (const std::exception&) {
            // Expected for invalid configurations
        }
        
        // 2. Test ConstantPad2d (expects 4D input: N, C, H, W)
        try {
            auto input_4d = input.dim() == 4 ? input :
                           (input.dim() < 4 ? input.view({1, 1, -1, 1}) : input.flatten().view({1, 1, -1, 1}));
            if (input_4d.numel() > 0) {
                std::vector<int64_t> pad4 = {padding[0] % 8, padding[1] % 8, 
                                              padding.size() > 2 ? padding[2] % 8 : padding[0] % 8,
                                              padding.size() > 3 ? padding[3] % 8 : padding[1] % 8};
                auto options = torch::nn::ConstantPad2dOptions(pad4, pad_value);
                auto module = torch::nn::ConstantPad2d(options);
                auto output = module->forward(input_4d);
            }
        } catch (const std::exception&) {
        }
        
        // 3. Test ConstantPad3d (expects 5D input: N, C, D, H, W)
        try {
            auto input_5d = input.dim() == 5 ? input :
                           (input.numel() > 0 ? input.flatten().view({1, 1, 1, 1, -1}) : torch::zeros({1, 1, 1, 1, 1}));
            std::vector<int64_t> pad6 = {padding[0] % 4, padding[1] % 4,
                                          padding.size() > 2 ? padding[2] % 4 : 0,
                                          padding.size() > 3 ? padding[3] % 4 : 0,
                                          padding.size() > 4 ? padding[4] % 4 : 0,
                                          padding.size() > 5 ? padding[5] % 4 : 0};
            auto options = torch::nn::ConstantPad3dOptions(pad6, pad_value);
            auto module = torch::nn::ConstantPad3d(options);
            auto output = module->forward(input_5d);
        } catch (const std::exception&) {
        }
        
        // 4. Test ReflectionPad1d (expects 3D input, input size must be >= padding)
        try {
            auto input_3d = input.dim() == 3 ? input : 
                           (input.numel() >= 4 ? input.flatten().narrow(0, 0, 4).view({1, 1, 4}) : torch::randn({1, 1, 8}));
            int64_t max_pad = std::max(1L, input_3d.size(-1) - 1);
            auto options = torch::nn::ReflectionPad1dOptions({padding[0] % max_pad, padding[1] % max_pad});
            auto module = torch::nn::ReflectionPad1d(options);
            auto output = module->forward(input_3d);
        } catch (const std::exception&) {
        }
        
        // 5. Test ReflectionPad2d (expects 4D input)
        try {
            auto input_4d = input.dim() == 4 ? input : torch::randn({1, 1, 8, 8});
            int64_t max_pad_h = std::max(1L, input_4d.size(-2) - 1);
            int64_t max_pad_w = std::max(1L, input_4d.size(-1) - 1);
            std::vector<int64_t> pad4 = {padding[0] % max_pad_w, padding[1] % max_pad_w,
                                          padding.size() > 2 ? padding[2] % max_pad_h : 0,
                                          padding.size() > 3 ? padding[3] % max_pad_h : 0};
            auto options = torch::nn::ReflectionPad2dOptions(pad4);
            auto module = torch::nn::ReflectionPad2d(options);
            auto output = module->forward(input_4d);
        } catch (const std::exception&) {
        }
        
        // 6. Test ReflectionPad3d (expects 5D input)
        try {
            auto input_5d = torch::randn({1, 1, 4, 4, 4});
            std::vector<int64_t> pad6 = {padding[0] % 3, padding[1] % 3,
                                          padding.size() > 2 ? padding[2] % 3 : 0,
                                          padding.size() > 3 ? padding[3] % 3 : 0,
                                          padding.size() > 4 ? padding[4] % 3 : 0,
                                          padding.size() > 5 ? padding[5] % 3 : 0};
            auto options = torch::nn::ReflectionPad3dOptions(pad6);
            auto module = torch::nn::ReflectionPad3d(options);
            auto output = module->forward(input_5d);
        } catch (const std::exception&) {
        }
        
        // 7. Test ReplicationPad1d
        try {
            auto input_3d = input.dim() == 3 ? input : 
                           (input.numel() > 0 ? input.flatten().unsqueeze(0).unsqueeze(0) : torch::randn({1, 1, 4}));
            auto options = torch::nn::ReplicationPad1dOptions({padding[0] % 8, padding[1] % 8});
            auto module = torch::nn::ReplicationPad1d(options);
            auto output = module->forward(input_3d);
        } catch (const std::exception&) {
        }
        
        // 8. Test ReplicationPad2d
        try {
            auto input_4d = input.dim() == 4 ? input : torch::randn({1, 1, 4, 4});
            std::vector<int64_t> pad4 = {padding[0] % 8, padding[1] % 8,
                                          padding.size() > 2 ? padding[2] % 8 : 0,
                                          padding.size() > 3 ? padding[3] % 8 : 0};
            auto options = torch::nn::ReplicationPad2dOptions(pad4);
            auto module = torch::nn::ReplicationPad2d(options);
            auto output = module->forward(input_4d);
        } catch (const std::exception&) {
        }
        
        // 9. Test ReplicationPad3d
        try {
            auto input_5d = input.dim() == 5 ? input : torch::randn({1, 1, 2, 2, 2});
            std::vector<int64_t> pad6 = {padding[0] % 4, padding[1] % 4,
                                          padding.size() > 2 ? padding[2] % 4 : 0,
                                          padding.size() > 3 ? padding[3] % 4 : 0,
                                          padding.size() > 4 ? padding[4] % 4 : 0,
                                          padding.size() > 5 ? padding[5] % 4 : 0};
            auto options = torch::nn::ReplicationPad3dOptions(pad6);
            auto module = torch::nn::ReplicationPad3d(options);
            auto output = module->forward(input_5d);
        } catch (const std::exception&) {
        }
        
        // 10. Test ZeroPad1d
        try {
            auto input_3d = input.dim() == 3 ? input : torch::randn({1, 1, 4});
            auto options = torch::nn::ZeroPad1dOptions({padding[0] % 8, padding[1] % 8});
            auto module = torch::nn::ZeroPad1d(options);
            auto output = module->forward(input_3d);
        } catch (const std::exception&) {
        }
        
        // 11. Test ZeroPad2d
        try {
            auto input_4d = input.dim() == 4 ? input : torch::randn({1, 1, 4, 4});
            std::vector<int64_t> pad4 = {padding[0] % 8, padding[1] % 8,
                                          padding.size() > 2 ? padding[2] % 8 : 0,
                                          padding.size() > 3 ? padding[3] % 8 : 0};
            auto options = torch::nn::ZeroPad2dOptions(pad4);
            auto module = torch::nn::ZeroPad2d(options);
            auto output = module->forward(input_4d);
        } catch (const std::exception&) {
        }
        
        // 12. Test ZeroPad3d
        try {
            auto input_5d = input.dim() == 5 ? input : torch::randn({1, 1, 2, 2, 2});
            std::vector<int64_t> pad6 = {padding[0] % 4, padding[1] % 4,
                                          padding.size() > 2 ? padding[2] % 4 : 0,
                                          padding.size() > 3 ? padding[3] % 4 : 0,
                                          padding.size() > 4 ? padding[4] % 4 : 0,
                                          padding.size() > 5 ? padding[5] % 4 : 0};
            auto options = torch::nn::ZeroPad3dOptions(pad6);
            auto module = torch::nn::ZeroPad3d(options);
            auto output = module->forward(input_5d);
        } catch (const std::exception&) {
        }
        
        // 13. Test functional pad API with constant mode
        try {
            auto input_4d = input.dim() >= 2 ? input.view({1, 1, -1, 1}) : torch::randn({1, 1, 4, 4});
            std::vector<int64_t> func_pad = {padding[0] % 4, padding[1] % 4};
            auto output = torch::nn::functional::pad(
                input_4d,
                torch::nn::functional::PadFuncOptions(func_pad).mode(torch::kConstant).value(pad_value)
            );
        } catch (const std::exception&) {
        }
        
        // 14. Test functional pad API with reflect mode
        try {
            auto input_4d = torch::randn({1, 1, 8, 8});
            int64_t max_pad = std::max(1L, std::min(input_4d.size(-1), input_4d.size(-2)) - 1);
            std::vector<int64_t> func_pad = {padding[0] % max_pad, padding[1] % max_pad};
            auto output = torch::nn::functional::pad(
                input_4d,
                torch::nn::functional::PadFuncOptions(func_pad).mode(torch::kReflect)
            );
        } catch (const std::exception&) {
        }
        
        // 15. Test functional pad API with replicate mode
        try {
            auto input_4d = torch::randn({1, 1, 4, 4});
            std::vector<int64_t> func_pad = {padding[0] % 4, padding[1] % 4};
            auto output = torch::nn::functional::pad(
                input_4d,
                torch::nn::functional::PadFuncOptions(func_pad).mode(torch::kReplicate)
            );
        } catch (const std::exception&) {
        }
        
        // 16. Test functional pad API with circular mode
        try {
            auto input_4d = torch::randn({1, 1, 8, 8});
            std::vector<int64_t> func_pad = {padding[0] % 4, padding[1] % 4};
            auto output = torch::nn::functional::pad(
                input_4d,
                torch::nn::functional::PadFuncOptions(func_pad).mode(torch::kCircular)
            );
        } catch (const std::exception&) {
        }
        
        // 17. Test functional pad with fuzzer-selected mode
        try {
            auto input_4d = torch::randn({1, 1, 8, 8});
            std::vector<int64_t> func_pad = {padding[0] % 4, padding[1] % 4};
            auto output = torch::nn::functional::pad(
                input_4d,
                torch::nn::functional::PadFuncOptions(func_pad).mode(padding_mode).value(pad_value)
            );
        } catch (const std::exception&) {
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}