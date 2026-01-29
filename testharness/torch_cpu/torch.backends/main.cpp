#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <torch/torch.h>
#include <ATen/Context.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    // Target API: torch.backends
    (void)"torch.backends";
    
    try
    {
        size_t offset = 0;
        
        // Skip if not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to use with backends operations
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test torch::cuda operations (backends.cuda)
        bool cuda_available = torch::cuda::is_available();
        if (cuda_available) {
            int device_count = torch::cuda::device_count();
            
            if (offset < Size && device_count > 0) {
                int device_idx = Data[offset++] % device_count;
                bool cudnn_available = torch::cuda::cudnn_is_available();
                if (cudnn_available) {
                    torch::cuda::manual_seed_all(static_cast<uint64_t>(Size));
                }
                
                // Exercise CUDA backend synchronization for a specific device
                torch::cuda::synchronize(device_idx);
            }
        }
        
        // Test global context settings (backends configuration)
        auto& context = at::globalContext();
        
        // Query backend availability - these are read-only checks
        bool mkldnn_enabled = context.userEnabledMkldnn();
        (void)mkldnn_enabled;
        
        // Test setting deterministic algorithms (backends.cudnn.deterministic equivalent)
        if (offset < Size) {
            bool set_deterministic = Data[offset++] % 2 == 0;
            try {
                at::globalContext().setDeterministicCuDNN(set_deterministic);
            } catch (const std::exception&) {
                // Ignore if not supported
            }
        }
        
        // Test benchmark mode (backends.cudnn.benchmark equivalent)
        if (offset < Size) {
            bool benchmark_mode = Data[offset++] % 2 == 0;
            try {
                at::globalContext().setBenchmarkCuDNN(benchmark_mode);
            } catch (const std::exception&) {
                // Ignore if not supported
            }
        }
        
        // Test allow_tf32 settings (backends.cuda.matmul.allow_tf32 equivalent)
        if (offset < Size) {
            bool allow_tf32 = Data[offset++] % 2 == 0;
            try {
                at::globalContext().setAllowTF32CuBLAS(allow_tf32);
                at::globalContext().setAllowTF32CuDNN(allow_tf32);
            } catch (const std::exception&) {
                // Ignore if not supported
            }
        }
        
        // Test basic tensor operations that exercise backends
        if (tensor.numel() > 0) {
            auto sum_result = tensor.sum();
            auto mean_result = tensor.mean();
            (void)sum_result;
            (void)mean_result;
            
            // Test different dtypes
            if (offset < Size) {
                auto dtype_idx = Data[offset++] % 4;
                torch::ScalarType target_dtype;
                switch (dtype_idx) {
                    case 0: target_dtype = torch::kFloat32; break;
                    case 1: target_dtype = torch::kFloat64; break;
                    case 2: target_dtype = torch::kInt32; break;
                    default: target_dtype = torch::kInt64; break;
                }
                try {
                    auto converted_tensor = tensor.to(target_dtype);
                    (void)converted_tensor;
                } catch (const std::exception&) {
                    // Ignore conversion errors
                }
            }
        }
        
        // Test device placement
        if (offset < Size) {
            bool use_cuda = Data[offset++] % 2 == 0;
            torch::Device target_device = use_cuda && torch::cuda::is_available() ? 
                torch::kCUDA : torch::kCPU;
            
            try {
                auto device_tensor = tensor.to(target_device);
                (void)device_tensor;
            } catch (const std::exception&) {
                // Ignore device placement errors
            }
        }
        
        // Test matmul operations that exercise different backends (MKL, BLAS)
        if (offset + 2 <= Size && tensor.dim() >= 1) {
            try {
                int m = (Data[offset++] % 16) + 1;
                int n = (Data[offset++] % 16) + 1;
                auto a = torch::randn({m, n}, torch::kFloat32);
                auto b = torch::randn({n, m}, torch::kFloat32);
                auto result = torch::matmul(a, b);
                (void)result;
            } catch (const std::exception&) {
                // Ignore matmul errors
            }
        }
        
        // Test convolution that exercises cudnn/mkldnn backends
        if (tensor.dim() == 4 && tensor.size(0) > 0 && tensor.size(1) > 0) {
            try {
                auto input = tensor.to(torch::kFloat32);
                int in_channels = input.size(1);
                int out_channels = (offset < Size) ? (Data[offset++] % 8) + 1 : 1;
                auto weight = torch::randn({out_channels, in_channels, 3, 3}, torch::kFloat32);
                auto conv_result = torch::conv2d(input, weight, {}, 1, 1);
                (void)conv_result;
            } catch (const std::exception&) {
                // Ignore convolution errors (shape mismatches, etc.)
            }
        }
        
        // Test autograd functionality
        if (tensor.requires_grad()) {
            try {
                auto grad_tensor = tensor.sum();
                grad_tensor.backward();
            } catch (const std::exception&) {
                // Ignore autograd errors
            }
        }
        
        // Test memory format operations
        if (tensor.dim() == 4 && offset < Size) {
            bool use_channels_last = Data[offset++] % 2 == 0;
            try {
                auto memory_format = use_channels_last ? 
                    torch::MemoryFormat::ChannelsLast : torch::MemoryFormat::Contiguous;
                auto formatted_tensor = tensor.contiguous(memory_format);
                (void)formatted_tensor;
            } catch (const std::exception&) {
                // Ignore memory format errors
            }
        }
        
        // Query flash SDP settings if available (backends.cuda.flash_sdp_enabled equivalent)
        // These are newer APIs, wrap in try-catch
        try {
            // Check various SDP backend preferences through global context
            bool cudnn_sdp = at::globalContext().userEnabledCuDNNSDP();
            bool flash_sdp = at::globalContext().userEnabledFlashSDP();
            bool math_sdp = at::globalContext().userEnabledMathSDP();
            (void)cudnn_sdp;
            (void)flash_sdp;
            (void)math_sdp;
        } catch (const std::exception&) {
            // Ignore if not available in this version
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}