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
        
        // Need at least a few bytes for the tensor and parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create the weight tensor
        torch::Tensor weight = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least one byte left for dimension selection
        if (offset >= Size) {
            return 0;
        }
        
        // Extract dimension parameter
        int64_t dim = static_cast<int64_t>(Data[offset++]) % (weight.dim() + 1);
        
        // Extract name parameter (use "weight" or "bias" based on a byte)
        std::string name = (offset < Size && Data[offset++] % 2 == 0) ? "weight" : "bias";
        
        // Create a simple module to apply weight_norm to
        struct TestModule : torch::nn::Module {
            TestModule(torch::Tensor weight) {
                register_parameter("original", weight);
            }
            
            torch::Tensor forward(torch::Tensor input) {
                return original;
            }
            
            torch::Tensor original;
        };
        
        auto module = std::make_shared<TestModule>(weight);
        
        // Apply weight_norm to the module using torch::nn::utils::weight_norm
        torch::nn::utils::weight_norm(*module, "original", dim);
        
        // Try to access the weight and v parameters created by weight_norm
        auto weight_param = module->named_parameters().find("original_v");
        auto g_param = module->named_parameters().find("original_g");
        
        // Try to compute the weight using the formula: weight = g * v / ||v||
        if (weight_param != module->named_parameters().end() && 
            g_param != module->named_parameters().end()) {
            auto v = weight_param->value();
            auto g = g_param->value();
            
            // Compute the norm of v along the specified dimension
            auto norm = torch::norm(v, 2, dim, true);
            
            // Compute the weight using the formula
            auto computed_weight = g * (v / norm);
            
            // Try to access the original weight
            auto original_weight = module->named_buffers().find("original");
            if (original_weight != module->named_buffers().end()) {
                // Compare the computed weight with the original weight
                auto diff = torch::abs(computed_weight - original_weight->value()).max().item<float>();
            }
        }
        
        // Try to remove weight_norm
        torch::nn::utils::remove_weight_norm(*module, "original");
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
