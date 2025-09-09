#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }

        auto tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        auto tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        tensor1.contiguous();
        tensor1.is_contiguous();
        tensor1.numel();
        tensor1.dim();
        tensor1.sizes();
        tensor1.strides();
        tensor1.dtype();
        tensor1.device();
        tensor1.layout();
        tensor1.requires_grad();
        tensor1.is_leaf();
        tensor1.grad_fn();
        tensor1.storage();
        tensor1.storage_offset();
        tensor1.itemsize();
        tensor1.element_size();
        tensor1.nbytes();
        tensor1.ndimension();
        tensor1.size(0);
        
        if (tensor1.dim() > 0) {
            tensor1.stride(0);
        }
        
        tensor1.is_cuda();
        tensor1.is_sparse();
        tensor1.is_mkldnn();
        tensor1.is_quantized();
        tensor1.is_meta();
        tensor1.is_complex();
        tensor1.is_floating_point();
        tensor1.is_signed();
        tensor1.has_names();
        tensor1.is_pinned();
        
        auto cloned = tensor1.clone();
        auto detached = tensor1.detach();
        auto copied = tensor1.to(tensor1.dtype());
        
        if (tensor1.numel() > 0) {
            tensor1.data_ptr();
        }
        
        if (tensor1.dtype() == torch::kFloat || tensor1.dtype() == torch::kDouble) {
            tensor1.sum();
            tensor1.mean();
            if (tensor1.numel() > 0) {
                tensor1.min();
                tensor1.max();
            }
        }
        
        if (tensor1.dtype() == torch::kBool) {
            tensor1.any();
            tensor1.all();
        }
        
        if (tensor1.sizes() == tensor2.sizes() && tensor1.dtype() == tensor2.dtype()) {
            tensor1 + tensor2;
            tensor1 - tensor2;
            tensor1 * tensor2;
            if (tensor2.dtype().isFloatingType()) {
                tensor1 / tensor2;
            }
            tensor1.add(tensor2);
            tensor1.sub(tensor2);
            tensor1.mul(tensor2);
            if (tensor2.dtype().isFloatingType()) {
                tensor1.div(tensor2);
            }
        }
        
        if (tensor1.dim() >= 2) {
            tensor1.transpose(0, 1);
            tensor1.t();
            tensor1.flatten();
        }
        
        if (tensor1.dim() == 1 && tensor1.numel() > 0) {
            tensor1.unsqueeze(0);
            tensor1.unsqueeze(-1);
        }
        
        if (tensor1.dim() > 1) {
            for (int64_t i = 0; i < tensor1.dim(); ++i) {
                if (tensor1.size(i) == 1) {
                    tensor1.squeeze(i);
                    break;
                }
            }
        }
        
        if (tensor1.dim() >= 2 && tensor1.size(0) > 0 && tensor1.size(1) > 0) {
            tensor1.view({-1});
            tensor1.reshape({tensor1.numel()});
        }
        
        if (tensor1.dtype().isFloatingType() && tensor1.numel() > 0) {
            tensor1.abs();
            tensor1.sqrt();
            tensor1.exp();
            tensor1.log();
            tensor1.sin();
            tensor1.cos();
            tensor1.tanh();
            tensor1.sigmoid();
            tensor1.relu();
        }
        
        if (tensor1.dtype().isIntegralType(false)) {
            tensor1.abs();
        }
        
        tensor1.zero_();
        tensor1.fill_(1.0);
        
        if (tensor1.numel() > 0) {
            tensor1[0];
        }
        
        if (tensor1.dim() >= 2 && tensor1.size(0) > 0) {
            tensor1[0];
        }
        
        if (tensor1.dim() >= 3 && tensor1.size(0) > 0 && tensor1.size(1) > 0) {
            tensor1[0][0];
        }
        
        auto narrow_tensor = tensor1;
        if (narrow_tensor.dim() > 0 && narrow_tensor.size(0) > 1) {
            narrow_tensor.narrow(0, 0, 1);
        }
        
        auto select_tensor = tensor1;
        if (select_tensor.dim() > 0 && select_tensor.size(0) > 0) {
            select_tensor.select(0, 0);
        }
        
        if (tensor1.dim() >= 2) {
            std::vector<torch::indexing::TensorIndex> indices;
            indices.push_back(torch::indexing::Slice(0, tensor1.size(0)));
            indices.push_back(torch::indexing::Slice(0, tensor1.size(1)));
            tensor1.index(indices);
        }
        
        if (tensor1.dtype().isFloatingType()) {
            tensor1.requires_grad_(true);
            if (tensor1.requires_grad()) {
                auto output = tensor1.sum();
                if (output.requires_grad()) {
                    output.backward();
                }
            }
        }
        
        tensor1.cpu();
        
        if (tensor1.dtype() != torch::kBool) {
            tensor1.to(torch::kFloat);
        }
        
        tensor1.to(torch::kCPU);
        
        auto permute_tensor = tensor1;
        if (permute_tensor.dim() >= 2) {
            std::vector<int64_t> dims;
            for (int64_t i = permute_tensor.dim() - 1; i >= 0; --i) {
                dims.push_back(i);
            }
            permute_tensor.permute(dims);
        }
        
        if (tensor1.numel() > 1) {
            tensor1.sort();
        }
        
        if (tensor1.dtype().isFloatingType() && tensor1.numel() > 0) {
            tensor1.isnan();
            tensor1.isinf();
            tensor1.isfinite();
        }
        
        tensor1.nonzero();
        
        if (tensor1.dtype() == torch::kBool) {
            tensor1.nonzero();
        }
        
        auto repeat_tensor = tensor1;
        if (repeat_tensor.dim() > 0) {
            std::vector<int64_t> repeats(repeat_tensor.dim(), 1);
            repeats[0] = 2;
            repeat_tensor.repeat(repeats);
        }
        
        if (tensor1.dim() > 0) {
            tensor1.expand({tensor1.size(0) * 2});
        }
        
        tensor1.contiguous();
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}