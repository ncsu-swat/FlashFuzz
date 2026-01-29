#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;
        
        // Use first bytes to select which lazy module to test
        uint8_t module_selector = Data[offset++] % 6;
        uint8_t batch_size = (Data[offset++] % 4) + 1;  // 1-4
        uint8_t channels = (Data[offset++] % 8) + 1;    // 1-8
        uint8_t spatial = (Data[offset++] % 4) + 2;     // 2-5
        uint8_t out_features = (Data[offset++] % 32) + 8; // 8-39

        torch::NoGradGuard no_grad;

        switch (module_selector) {
            case 0: {
                // Test LazyLinear
                auto lazy_linear = torch::nn::LazyLinear(out_features);
                int64_t in_features = (Data[offset % Size] % 16) + 4;
                auto input = torch::randn({batch_size, in_features});
                try {
                    auto output = lazy_linear->forward(input);
                    // Verify output shape
                    assert(output.size(1) == out_features);
                } catch (const c10::Error& e) {
                    // Expected for invalid configurations
                }
                break;
            }
            case 1: {
                // Test LazyConv1d - requires 3D input (N, C, L)
                auto lazy_conv1d = torch::nn::LazyConv1d(
                    torch::nn::LazyConv1dOptions(out_features, 3).padding(1));
                auto input = torch::randn({batch_size, channels, spatial * 4});
                try {
                    auto output = lazy_conv1d->forward(input);
                } catch (const c10::Error& e) {
                    // Expected for invalid configurations
                }
                break;
            }
            case 2: {
                // Test LazyConv2d - requires 4D input (N, C, H, W)
                auto lazy_conv2d = torch::nn::LazyConv2d(
                    torch::nn::LazyConv2dOptions(out_features, 3).padding(1));
                auto input = torch::randn({batch_size, channels, spatial, spatial});
                try {
                    auto output = lazy_conv2d->forward(input);
                } catch (const c10::Error& e) {
                    // Expected for invalid configurations
                }
                break;
            }
            case 3: {
                // Test LazyConv3d - requires 5D input (N, C, D, H, W)
                auto lazy_conv3d = torch::nn::LazyConv3d(
                    torch::nn::LazyConv3dOptions(out_features, 3).padding(1));
                auto input = torch::randn({batch_size, channels, spatial, spatial, spatial});
                try {
                    auto output = lazy_conv3d->forward(input);
                } catch (const c10::Error& e) {
                    // Expected for invalid configurations
                }
                break;
            }
            case 4: {
                // Test LazyBatchNorm1d - requires 2D or 3D input
                auto lazy_bn1d = torch::nn::LazyBatchNorm1d();
                auto input = torch::randn({batch_size, channels, spatial * 2});
                try {
                    lazy_bn1d->train();
                    auto output = lazy_bn1d->forward(input);
                } catch (const c10::Error& e) {
                    // Expected for invalid configurations
                }
                break;
            }
            case 5: {
                // Test LazyBatchNorm2d - requires 4D input
                auto lazy_bn2d = torch::nn::LazyBatchNorm2d();
                auto input = torch::randn({batch_size, channels, spatial, spatial});
                try {
                    lazy_bn2d->train();
                    auto output = lazy_bn2d->forward(input);
                } catch (const c10::Error& e) {
                    // Expected for invalid configurations
                }
                break;
            }
        }

        // Additional test: LazyConvTranspose2d
        if (Size > offset + 4) {
            uint8_t selector2 = Data[offset++] % 3;
            if (selector2 == 0) {
                auto lazy_convt2d = torch::nn::LazyConvTranspose2d(
                    torch::nn::LazyConvTranspose2dOptions(out_features, 3).padding(1));
                auto input = torch::randn({batch_size, channels, spatial, spatial});
                try {
                    auto output = lazy_convt2d->forward(input);
                } catch (const c10::Error& e) {
                    // Expected for invalid configurations
                }
            } else if (selector2 == 1) {
                // Test LazyInstanceNorm2d
                auto lazy_in2d = torch::nn::LazyInstanceNorm2d();
                auto input = torch::randn({batch_size, channels, spatial, spatial});
                try {
                    auto output = lazy_in2d->forward(input);
                } catch (const c10::Error& e) {
                    // Expected for invalid configurations
                }
            } else {
                // Test parameter materialization
                auto lazy_linear = torch::nn::LazyLinear(out_features);
                auto input = torch::randn({batch_size, channels});
                try {
                    auto output = lazy_linear->forward(input);
                    // Access parameters after materialization
                    auto params = lazy_linear->parameters();
                    for (auto& param : params) {
                        auto shape = param.sizes();
                        auto numel = param.numel();
                        (void)numel;
                    }
                } catch (const c10::Error& e) {
                    // Expected for invalid configurations
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}