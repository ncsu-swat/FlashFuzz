#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract padding parameters from the remaining data
        uint8_t padding_mode_byte = 0;
        if (offset < Size) {
            padding_mode_byte = Data[offset++];
        }
        
        // Select padding mode
        std::string padding_mode;
        switch (padding_mode_byte % 3) {
            case 0:
                padding_mode = "constant";
                break;
            case 1:
                padding_mode = "reflect";
                break;
            case 2:
                padding_mode = "replicate";
                break;
        }
        
        // Extract padding values
        std::vector<int64_t> padding;
        uint8_t padding_size = 0;
        if (offset < Size) {
            padding_size = Data[offset++] % 6 + 1; // 1-6 padding values
        }
        
        for (uint8_t i = 0; i < padding_size && offset + sizeof(int16_t) <= Size; i++) {
            int16_t pad_value;
            std::memcpy(&pad_value, Data + offset, sizeof(int16_t));
            offset += sizeof(int16_t);
            padding.push_back(static_cast<int64_t>(pad_value));
        }
        
        // If no padding values were extracted, use default
        if (padding.empty()) {
            padding.push_back(0);
        }
        
        // Test different padding modules
        
        // 1. Test ConstantPad1d
        if (input.dim() >= 1) {
            try {
                auto pad_size = padding.size() >= 2 ? 
                    std::vector<int64_t>{padding[0], padding[1]} : 
                    std::vector<int64_t>{padding[0], padding[0]};
                
                auto constantpad1d = torch::nn::ConstantPad1d(pad_size, 0.0);
                auto output1 = constantpad1d->forward(input);
            } catch (const std::exception&) {
                // Allow exceptions from the operation
            }
        }
        
        // 2. Test ConstantPad2d
        if (input.dim() >= 2) {
            try {
                auto pad_size = padding.size() >= 4 ? 
                    std::vector<int64_t>{padding[0], padding[1], padding[2], padding[3]} : 
                    std::vector<int64_t>{padding[0], padding[0], padding[0], padding[0]};
                
                auto constantpad2d = torch::nn::ConstantPad2d(pad_size, 0.0);
                auto output2 = constantpad2d->forward(input);
            } catch (const std::exception&) {
                // Allow exceptions from the operation
            }
        }
        
        // 3. Test ConstantPad3d
        if (input.dim() >= 3) {
            try {
                auto pad_size = padding.size() >= 6 ? 
                    std::vector<int64_t>{padding[0], padding[1], padding[2], padding[3], padding[4], padding[5]} : 
                    std::vector<int64_t>{padding[0], padding[0], padding[0], padding[0], padding[0], padding[0]};
                
                auto constantpad3d = torch::nn::ConstantPad3d(pad_size, 0.0);
                auto output3 = constantpad3d->forward(input);
            } catch (const std::exception&) {
                // Allow exceptions from the operation
            }
        }
        
        // 4. Test ReflectionPad1d
        if (input.dim() >= 1) {
            try {
                auto pad_size = padding.size() >= 2 ? 
                    std::vector<int64_t>{padding[0], padding[1]} : 
                    std::vector<int64_t>{padding[0], padding[0]};
                
                auto reflectionpad1d = torch::nn::ReflectionPad1d(pad_size);
                auto output4 = reflectionpad1d->forward(input);
            } catch (const std::exception&) {
                // Allow exceptions from the operation
            }
        }
        
        // 5. Test ReflectionPad2d
        if (input.dim() >= 2) {
            try {
                auto pad_size = padding.size() >= 4 ? 
                    std::vector<int64_t>{padding[0], padding[1], padding[2], padding[3]} : 
                    std::vector<int64_t>{padding[0], padding[0], padding[0], padding[0]};
                
                auto reflectionpad2d = torch::nn::ReflectionPad2d(pad_size);
                auto output5 = reflectionpad2d->forward(input);
            } catch (const std::exception&) {
                // Allow exceptions from the operation
            }
        }
        
        // 6. Test ReplicationPad1d
        if (input.dim() >= 1) {
            try {
                auto pad_size = padding.size() >= 2 ? 
                    std::vector<int64_t>{padding[0], padding[1]} : 
                    std::vector<int64_t>{padding[0], padding[0]};
                
                auto replicationpad1d = torch::nn::ReplicationPad1d(pad_size);
                auto output6 = replicationpad1d->forward(input);
            } catch (const std::exception&) {
                // Allow exceptions from the operation
            }
        }
        
        // 7. Test ReplicationPad2d
        if (input.dim() >= 2) {
            try {
                auto pad_size = padding.size() >= 4 ? 
                    std::vector<int64_t>{padding[0], padding[1], padding[2], padding[3]} : 
                    std::vector<int64_t>{padding[0], padding[0], padding[0], padding[0]};
                
                auto replicationpad2d = torch::nn::ReplicationPad2d(pad_size);
                auto output7 = replicationpad2d->forward(input);
            } catch (const std::exception&) {
                // Allow exceptions from the operation
            }
        }
        
        // 8. Test ReplicationPad3d
        if (input.dim() >= 3) {
            try {
                auto pad_size = padding.size() >= 6 ? 
                    std::vector<int64_t>{padding[0], padding[1], padding[2], padding[3], padding[4], padding[5]} : 
                    std::vector<int64_t>{padding[0], padding[0], padding[0], padding[0], padding[0], padding[0]};
                
                auto replicationpad3d = torch::nn::ReplicationPad3d(pad_size);
                auto output8 = replicationpad3d->forward(input);
            } catch (const std::exception&) {
                // Allow exceptions from the operation
            }
        }
        
        // 9. Test ZeroPad2d
        if (input.dim() >= 2) {
            try {
                auto pad_size = padding.size() >= 4 ? 
                    std::vector<int64_t>{padding[0], padding[1], padding[2], padding[3]} : 
                    std::vector<int64_t>{padding[0], padding[0], padding[0], padding[0]};
                
                auto zeropad2d = torch::nn::ZeroPad2d(pad_size);
                auto output9 = zeropad2d->forward(input);
            } catch (const std::exception&) {
                // Allow exceptions from the operation
            }
        }
        
        // 10. Test functional padding
        try {
            auto output10 = torch::pad(
                input, 
                padding, 
                padding_mode,
                0.0
            );
        } catch (const std::exception&) {
            // Allow exceptions from the operation
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}