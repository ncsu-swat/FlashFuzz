#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 3) {
            return 0;
        }
        
        // Parse device type (CPU or CUDA)
        bool use_cuda = Data[offset++] % 2 == 1 && torch::cuda::is_available();
        torch::Device device = use_cuda ? torch::kCUDA : torch::kCPU;
        
        // Parse dtype for autocast
        uint8_t dtype_selector = Data[offset++];
        torch::ScalarType cast_dtype;
        if (dtype_selector % 3 == 0) {
            cast_dtype = torch::kFloat16;
        } else if (dtype_selector % 3 == 1) {
            cast_dtype = torch::kBFloat16;
        } else {
            cast_dtype = torch::kFloat32;
        }
        
        // Parse enabled flag
        bool enabled = Data[offset++] % 2 == 1;
        
        // Create input tensor
        torch::Tensor input;
        if (offset < Size) {
            input = fuzzer_utils::createTensor(Data, Size, offset);
            input = input.to(device);
        } else {
            // Create a default tensor if we don't have enough data
            input = torch::rand({2, 3}, torch::TensorOptions().device(device));
        }
        
        // Parse cache_enabled flag if we have more data
        bool cache_enabled = true;
        if (offset < Size) {
            cache_enabled = Data[offset++] % 2 == 1;
        }
        
        // Test autocast in different ways
        
        // 1. Using AutocastMode
        {
            c10::impl::AutocastState state(device.type(), enabled, cast_dtype, cache_enabled);
            torch::Tensor result = input * 2.0;
            torch::Tensor result2 = torch::matmul(result, result.transpose(0, 1));
            torch::Tensor result3 = torch::nn::functional::relu(result2);
        }
        
        // 2. Using AutocastMode with different operations
        {
            c10::impl::AutocastState state(device.type(), enabled, cast_dtype, cache_enabled);
            torch::Tensor result = input * 3.0;
            torch::Tensor result2 = torch::matmul(result, result.transpose(0, 1));
            torch::Tensor result3 = torch::nn::functional::relu(result2);
        }
        
        // 3. Check autocast state
        {
            c10::impl::AutocastState state(device.type(), enabled, cast_dtype, cache_enabled);
            bool is_enabled = c10::impl::is_autocast_enabled(device.type());
            
            torch::Tensor result = input * 4.0;
            torch::Tensor result2 = torch::matmul(result, result.transpose(0, 1));
        }
        
        // 4. Nested autocast
        {
            c10::impl::AutocastState outer_state(device.type(), enabled, cast_dtype, cache_enabled);
            torch::Tensor outer_result = input * 5.0;
            
            // Inner autocast with different dtype
            torch::ScalarType inner_dtype = (cast_dtype == torch::kFloat16) ? torch::kBFloat16 : torch::kFloat16;
            c10::impl::AutocastState inner_state(device.type(), !enabled, inner_dtype, !cache_enabled);
            torch::Tensor inner_result = outer_result * 2.0;
            
            // After inner state is destroyed, should return to outer settings
            torch::Tensor final_result = inner_result + outer_result;
        }
        
        // 5. Test with different tensor operations under autocast
        {
            c10::impl::AutocastState state(device.type(), enabled, cast_dtype, cache_enabled);
            torch::Tensor result1 = torch::add(input, input);
            torch::Tensor result2 = torch::mul(result1, 2.0);
            torch::Tensor result3 = torch::sigmoid(result2);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
