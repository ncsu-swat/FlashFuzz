#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Initialize process group
        auto file_store = c10::make_intrusive<c10d::FileStore>("/tmp/fuzzer_test", 1);
        auto options = c10d::ProcessGroupGloo::Options::create();
        options->timeout = std::chrono::milliseconds(10000);
        options->devices.push_back(c10::make_intrusive<c10d::ProcessGroupGloo::Device>(c10d::ProcessGroupGloo::createDeviceForHostname("127.0.0.1")));
        auto pg = c10::make_intrusive<c10d::ProcessGroupGloo>(file_store, 0, 1, options);
        
        // Create input tensor
        torch::Tensor tensor;
        if (offset < Size) {
            tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Create a second tensor if there's more data
        torch::Tensor tensor2;
        if (offset + 2 < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            tensor2 = tensor.clone();
        }
        
        // Get operation type from input data
        uint8_t op_type = 0;
        if (offset < Size) {
            op_type = Data[offset++] % 5;
        }
        
        // Try different distributed operations based on op_type
        switch (op_type) {
            case 0: {
                // Allreduce
                std::vector<torch::Tensor> tensors = {tensor};
                std::vector<std::vector<torch::Tensor>> output_tensors = {tensors};
                
                uint8_t reduce_op = 0;
                if (offset < Size) {
                    reduce_op = Data[offset++] % 4;
                }
                
                c10d::ReduceOp::RedOpType redOp;
                switch (reduce_op) {
                    case 0: redOp = c10d::ReduceOp::SUM; break;
                    case 1: redOp = c10d::ReduceOp::PRODUCT; break;
                    case 2: redOp = c10d::ReduceOp::MIN; break;
                    case 3: redOp = c10d::ReduceOp::MAX; break;
                    default: redOp = c10d::ReduceOp::SUM;
                }
                
                pg->allreduce(output_tensors, c10d::AllreduceOptions().reduceOp(redOp));
                break;
            }
            
            case 1: {
                // Broadcast
                std::vector<torch::Tensor> tensors = {tensor};
                std::vector<std::vector<torch::Tensor>> output_tensors = {tensors};
                
                int root_rank = 0;
                pg->broadcast(output_tensors, c10d::BroadcastOptions().rootRank(root_rank));
                break;
            }
            
            case 2: {
                // Allgather
                std::vector<torch::Tensor> tensors = {tensor};
                std::vector<std::vector<torch::Tensor>> output_tensors = {tensors};
                std::vector<std::vector<torch::Tensor>> output_lists = {{tensor2}};
                
                pg->allgather(output_lists, output_tensors);
                break;
            }
            
            case 3: {
                // Reduce
                std::vector<torch::Tensor> tensors = {tensor};
                std::vector<std::vector<torch::Tensor>> output_tensors = {tensors};
                
                uint8_t reduce_op = 0;
                if (offset < Size) {
                    reduce_op = Data[offset++] % 4;
                }
                
                c10d::ReduceOp::RedOpType redOp;
                switch (reduce_op) {
                    case 0: redOp = c10d::ReduceOp::SUM; break;
                    case 1: redOp = c10d::ReduceOp::PRODUCT; break;
                    case 2: redOp = c10d::ReduceOp::MIN; break;
                    case 3: redOp = c10d::ReduceOp::MAX; break;
                    default: redOp = c10d::ReduceOp::SUM;
                }
                
                int root_rank = 0;
                pg->reduce(output_tensors, c10d::ReduceOptions().reduceOp(redOp).rootRank(root_rank));
                break;
            }
            
            case 4: {
                // Scatter
                std::vector<torch::Tensor> tensors = {tensor};
                std::vector<std::vector<torch::Tensor>> output_tensors = {tensors};
                std::vector<std::vector<torch::Tensor>> input_lists = {{tensor2}};
                
                int root_rank = 0;
                pg->scatter(output_tensors, input_lists, c10d::ScatterOptions().rootRank(root_rank));
                break;
            }
            
            default:
                break;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
