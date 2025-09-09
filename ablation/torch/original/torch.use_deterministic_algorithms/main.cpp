#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        bool mode = Data[offset++] & 1;
        bool warn_only = Data[offset++] & 1;
        
        torch::use_deterministic_algorithms(mode, warn_only);
        
        if (offset < Size) {
            auto tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (offset < Size) {
                auto tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
                
                try {
                    auto result = torch::mm(tensor1, tensor2);
                } catch (...) {
                }
                
                try {
                    auto result = torch::bmm(tensor1, tensor2);
                } catch (...) {
                }
                
                try {
                    auto result = torch::mv(tensor1, tensor2.flatten());
                } catch (...) {
                }
            }
            
            try {
                auto result = torch::kthvalue(tensor1, 1);
            } catch (...) {
            }
            
            try {
                auto result = torch::median(tensor1);
            } catch (...) {
            }
            
            try {
                auto result = torch::histc(tensor1);
            } catch (...) {
            }
            
            try {
                auto result = torch::cumsum(tensor1, 0);
            } catch (...) {
            }
            
            if (tensor1.requires_grad()) {
                try {
                    auto pool = torch::nn::AvgPool3d(torch::nn::AvgPool3dOptions(2));
                    auto result = pool(tensor1);
                    result.sum().backward();
                } catch (...) {
                }
                
                try {
                    auto pool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({2, 2}));
                    auto result = pool(tensor1);
                    result.sum().backward();
                } catch (...) {
                }
                
                try {
                    auto pool = torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions(2));
                    auto result = pool(tensor1);
                    result.sum().backward();
                } catch (...) {
                }
                
                try {
                    auto pad = torch::nn::ReflectionPad1d(torch::nn::ReflectionPad1dOptions(1));
                    auto result = pad(tensor1);
                    result.sum().backward();
                } catch (...) {
                }
                
                try {
                    auto pad = torch::nn::ReplicationPad2d(torch::nn::ReplicationPad2dOptions(1));
                    auto result = pad(tensor1);
                    result.sum().backward();
                } catch (...) {
                }
            }
            
            try {
                auto conv = torch::nn::Conv1d(torch::nn::Conv1dOptions(1, 1, 3));
                auto result = conv(tensor1);
            } catch (...) {
            }
            
            try {
                auto conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 1, 3));
                auto result = conv(tensor1);
            } catch (...) {
            }
            
            try {
                auto conv = torch::nn::ConvTranspose1d(torch::nn::ConvTranspose1dOptions(1, 1, 3));
                auto result = conv(tensor1);
            } catch (...) {
            }
            
            if (offset < Size) {
                try {
                    auto indices = fuzzer_utils::createTensor(Data, Size, offset);
                    auto result = torch::gather(tensor1, 0, indices);
                } catch (...) {
                }
                
                try {
                    auto indices = fuzzer_utils::createTensor(Data, Size, offset);
                    auto result = torch::index_add(tensor1, 0, indices, tensor1);
                } catch (...) {
                }
                
                try {
                    auto indices = fuzzer_utils::createTensor(Data, Size, offset);
                    auto result = torch::index_select(tensor1, 0, indices);
                } catch (...) {
                }
                
                try {
                    auto values = fuzzer_utils::createTensor(Data, Size, offset);
                    tensor1.put_(indices, values, false);
                } catch (...) {
                }
                
                try {
                    auto values = fuzzer_utils::createTensor(Data, Size, offset);
                    tensor1.put_(indices, values, true);
                } catch (...) {
                }
                
                try {
                    auto src = fuzzer_utils::createTensor(Data, Size, offset);
                    tensor1.scatter_add_(0, indices, src);
                } catch (...) {
                }
                
                try {
                    auto src = fuzzer_utils::createTensor(Data, Size, offset);
                    tensor1.scatter_(0, indices, src);
                } catch (...) {
                }
                
                try {
                    auto src = fuzzer_utils::createTensor(Data, Size, offset);
                    tensor1.scatter_reduce_(0, indices, src, "sum");
                } catch (...) {
                }
                
                try {
                    auto src = fuzzer_utils::createTensor(Data, Size, offset);
                    tensor1.scatter_reduce_(0, indices, src, "prod");
                } catch (...) {
                }
                
                try {
                    auto src = fuzzer_utils::createTensor(Data, Size, offset);
                    tensor1.index_copy_(0, indices, src);
                } catch (...) {
                }
                
                try {
                    tensor1.index_put_({indices}, values, false);
                } catch (...) {
                }
                
                try {
                    tensor1.index_put_({indices}, values, true);
                } catch (...) {
                }
            }
            
            try {
                auto repeats = torch::ones({tensor1.size(0)}, torch::kLong);
                auto result = torch::repeat_interleave(tensor1, repeats, 0);
            } catch (...) {
            }
            
            if (offset < Size) {
                try {
                    auto weights = fuzzer_utils::createTensor(Data, Size, offset);
                    auto result = torch::bincount(tensor1, weights);
                } catch (...) {
                }
            }
            
            try {
                auto embedding = torch::nn::EmbeddingBag(torch::nn::EmbeddingBagOptions(10, 5).mode(torch::kMax));
                auto input = torch::randint(0, 10, {5}, torch::kLong);
                auto result = embedding(input);
                if (result.requires_grad()) {
                    result.sum().backward();
                }
            } catch (...) {
            }
            
            try {
                auto nll = torch::nn::NLLLoss();
                auto target = torch::randint(0, 5, {tensor1.size(0)}, torch::kLong);
                auto result = nll(tensor1, target);
            } catch (...) {
            }
            
            try {
                auto ctc = torch::nn::CTCLoss();
                auto targets = torch::randint(1, 5, {tensor1.size(0)}, torch::kLong);
                auto input_lengths = torch::full({tensor1.size(0)}, tensor1.size(1), torch::kLong);
                auto target_lengths = torch::full({tensor1.size(0)}, targets.size(0), torch::kLong);
                auto result = ctc(tensor1, targets, input_lengths, target_lengths);
                if (result.requires_grad()) {
                    result.backward();
                }
            } catch (...) {
            }
            
            try {
                auto grid = torch::rand({1, tensor1.size(-2), tensor1.size(-1), 2});
                auto result = torch::nn::functional::grid_sample(tensor1, grid);
                if (result.requires_grad()) {
                    result.sum().backward();
                }
            } catch (...) {
            }
            
            try {
                auto unpool = torch::nn::MaxUnpool1d(torch::nn::MaxUnpool1dOptions(2));
                auto indices = torch::randint(0, 2, tensor1.sizes(), torch::kLong);
                auto result = unpool(tensor1, indices);
            } catch (...) {
            }
            
            try {
                auto unpool = torch::nn::MaxUnpool2d(torch::nn::MaxUnpool2dOptions(2));
                auto indices = torch::randint(0, 4, tensor1.sizes(), torch::kLong);
                auto result = unpool(tensor1, indices);
            } catch (...) {
            }
            
            try {
                auto unpool = torch::nn::MaxUnpool3d(torch::nn::MaxUnpool3dOptions(2));
                auto indices = torch::randint(0, 8, tensor1.sizes(), torch::kLong);
                auto result = unpool(tensor1, indices);
            } catch (...) {
            }
            
            try {
                auto result = torch::nn::functional::interpolate(tensor1, 
                    torch::nn::functional::InterpolateFuncOptions().mode(torch::kLinear));
                if (result.requires_grad()) {
                    result.sum().backward();
                }
            } catch (...) {
            }
            
            try {
                auto result = torch::nn::functional::interpolate(tensor1, 
                    torch::nn::functional::InterpolateFuncOptions().mode(torch::kBilinear));
                if (result.requires_grad()) {
                    result.sum().backward();
                }
            } catch (...) {
            }
            
            try {
                auto result = torch::nn::functional::interpolate(tensor1, 
                    torch::nn::functional::InterpolateFuncOptions().mode(torch::kBicubic));
                if (result.requires_grad()) {
                    result.sum().backward();
                }
            } catch (...) {
            }
            
            try {
                auto result = torch::nn::functional::interpolate(tensor1, 
                    torch::nn::functional::InterpolateFuncOptions().mode(torch::kTrilinear));
                if (result.requires_grad()) {
                    result.sum().backward();
                }
            } catch (...) {
            }
            
            try {
                auto maxpool = torch::nn::AdaptiveMaxPool2d(torch::nn::AdaptiveMaxPool2dOptions({2, 2}));
                auto result = maxpool(tensor1);
                if (result.requires_grad()) {
                    result.sum().backward();
                }
            } catch (...) {
            }
            
            try {
                auto fracpool = torch::nn::FractionalMaxPool2d(torch::nn::FractionalMaxPool2dOptions(2));
                auto result = fracpool(tensor1);
                if (result.requires_grad()) {
                    result.sum().backward();
                }
            } catch (...) {
            }
            
            try {
                auto fracpool = torch::nn::FractionalMaxPool3d(torch::nn::FractionalMaxPool3dOptions(2));
                auto result = fracpool(tensor1);
                if (result.requires_grad()) {
                    result.sum().backward();
                }
            } catch (...) {
            }
            
            try {
                auto pad = torch::nn::ReflectionPad2d(torch::nn::ReflectionPad2dOptions(1));
                auto result = pad(tensor1);
                if (result.requires_grad()) {
                    result.sum().backward();
                }
            } catch (...) {
            }
            
            try {
                auto pad = torch::nn::ReflectionPad3d(torch::nn::ReflectionPad3dOptions(1));
                auto result = pad(tensor1);
                if (result.requires_grad()) {
                    result.sum().backward();
                }
            } catch (...) {
            }
            
            try {
                auto pad = torch::nn::ReplicationPad1d(torch::nn::ReplicationPad1dOptions(1));
                auto result = pad(tensor1);
                if (result.requires_grad()) {
                    result.sum().backward();
                }
            } catch (...) {
            }
            
            try {
                auto pad = torch::nn::ReplicationPad3d(torch::nn::ReplicationPad3dOptions(1));
                auto result = pad(tensor1);
                if (result.requires_grad()) {
                    result.sum().backward();
                }
            } catch (...) {
            }
            
            try {
                if (tensor1.is_quantized()) {
                    tensor1.resize_({tensor1.numel()});
                }
            } catch (...) {
            }
        }
        
        torch::use_deterministic_algorithms(false, false);
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}