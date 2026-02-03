#include <torch/script.h>

#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <ATen/core/jit_type.h>  // for c10::TensorType::createContiguous

#include <ATen/ATen.h>

#include <iostream>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <cmath>
#include <string>

static size_t scalar_type_size(c10::ScalarType t) {
  return c10::elementSize(t); // bytes per element
}

// tensor shape and dtype => # of bytes
static size_t tensor_type_nbytes(const c10::VaryingShape<int64_t>& sizes, c10::ScalarType st) {
  auto cs = sizes.concrete_sizes();
  if (!cs.has_value()) return 0;

  const auto& dims = cs.value(); // std::vector<int64_t>
  int64_t n = 1;
  for (auto d : dims)
    n *= d;

  return static_cast<size_t>(n) * scalar_type_size(st);
}

static std::string bytes_to_mb(size_t n){
  double mb = static_cast<double>(n) / (1024.0 * 1024.0);
  return std::to_string(mb);
}

/*
  Compute liveness of tensor : where does a tensor die???
*/
static std::unordered_map<const torch::jit::Value*, size_t> compute_last_use_indices(const std::vector<torch::jit::Node*>& topo, const std::unordered_map<torch::jit::Node*, size_t>& nodeIdx) {
  std::unordered_map<const torch::jit::Value*, size_t> lastUse;
  for (size_t i = 0; i < topo.size(); ++i) {

    auto* node = topo[i];
    for (auto* val : node->outputs()) {

      if (val->uses().empty()) {
        lastUse[val] = i;

      } else {

        size_t maxIdx = i;
        for (const auto& usg : val->uses()) {

          auto it = nodeIdx.find(usg.user);
          if (it != nodeIdx.end() && it->second > maxIdx)
            maxIdx = it->second;
        }

        lastUse[val] = maxIdx;
      }
    }
  }

  return lastUse;
}

int main() {
  try {
    // const std::string ts_path = "../../data/alexnetb1_traced.pt";
    // const std::string ts_path = "../../data/mobilenet_v3_smallb1_traced.pt";
    // const std::string ts_path = "../../data/resnet50b1_traced.pt";
    // const std::string ts_path = "../../data/resnet101b1_traced.pt";
    const std::string ts_path = "../../data/vgg16b1_traced.pt";
    torch::jit::Module module = torch::jit::load(ts_path);

    size_t paramBytes = 0;
    for (const auto& p : module.named_parameters(/*recurse=*/true)) {
      const auto& t = p.value;
      paramBytes += (size_t)t.numel() * t.element_size();
    }

    for (const auto& b : module.named_buffers(/*recurse=*/true)) {
      const auto& t = b.value;
      paramBytes += (size_t)t.numel() * t.element_size();
    }

    module = torch::jit::freeze_module(std::move(module));

    auto method = module.get_method("forward");
    std::shared_ptr<torch::jit::Graph> graph = method.graph();

    torch::jit::Inline(*graph);                 // Done in freeze part so redundant
    // torch::jit::ConstantPropagation(graph);     // Done in freeze part so redundant

    int64_t B = 1, C = 3, H = 224, W = 224;

    // Find the first Tensor input (skip %self if present)
    torch::jit::Value* data_in = nullptr;
    for (auto* in : graph->inputs()) {
      if (in->type()->cast<c10::TensorType>()) {
          data_in = in;
          break;
      }
    }

    if (!data_in) {
      std::cerr << "No Tensor input found in graph->inputs().\n";
      return 1;
    }

    at::Device dev(at::kCPU);

    // This sets dtype + device + *concrete sizes* (+ contiguous strides implied)
    auto input_tt = c10::TensorType::createContiguous(at::kFloat, dev, {B, C, H, W});
    data_in->setType(input_tt);

    // Optional: print input type to verify it stuck
    // std::cout << "Annotated input %" << data_in->debugName()
    //           << " type = " << data_in->type()->str() << "\n";

    // run shape propagation ----
    torch::jit::PropagateInputShapes(graph);

    std::vector<torch::jit::Node*> topo;
    std::unordered_map<torch::jit::Node*, size_t> nodeIdx;
    size_t idx = 0;
    for (auto* n : graph->nodes()) {
      topo.push_back(n);
      nodeIdx[n] = idx++;
    }

    auto lastUse = compute_last_use_indices(topo, nodeIdx);

    size_t liveBytes = 0;         // running total of all live active bytes
    size_t peakActBytes = 0;      // peak active liveBytes ever reaches

    struct LiveVal {
      const torch::jit::Value* v; // SSA
      size_t bytes;               // size in bytes
      size_t dieAt;               // where it dies
    };
    std::vector<LiveVal> liveSet;

    for (size_t i = 0; i < topo.size(); ++i) {
      auto* n = topo[i];

      // 1) Skip nodes that correspond to weights/constants or attribute fetch.
      //    After freeze, parameters/attrs may be inlined as prim::Constant tensors.
      if (n->kind() == c10::prim::GetAttr) continue;
      if (n->kind() == c10::prim::Constant) continue;

      for (auto* v : n->outputs()) { 
        auto tensorType = v->type()->cast<c10::TensorType>();
        if (!tensorType) continue; // Skip non-tensor outputs (shapes, bool flags) -> doesn't need memory allocation

        auto tensorSt = tensorType->scalarType();
        if (!tensorSt.has_value()) continue;    // skip unknown dtype

        size_t bytes = tensor_type_nbytes(tensorType->sizes(), *tensorSt);
        if (bytes == 0) continue; // skiip unknown shape (TensorType(shape=[?, ?, ?, ?], dtype=float)) could be error

        size_t dieAt = lastUse.count(v) ? lastUse[v] : i; // default is die now (ex last node)
        liveSet.push_back({v, bytes, dieAt});
        liveBytes += bytes;

        if (liveBytes > peakActBytes) peakActBytes = liveBytes;
      }

      // free values whose last use is this node
      // To prevent case where a node's output and its input overlap free first
      for (auto it = liveSet.begin(); it != liveSet.end();) {
        if (it->dieAt == i) {
          liveBytes -= it->bytes;
          it = liveSet.erase(it);

        } else {
          ++it;
        }
      }
    }

    const size_t totalPeak = paramBytes + peakActBytes;
    std::cout << "Parameter bytes(MB):        " << bytes_to_mb(paramBytes) << "\n";
    std::cout << "Peak activations(MB):       " << bytes_to_mb(peakActBytes) << "\n";
    std::cout << "Total peak (inference):     " << bytes_to_mb(totalPeak) << "\n";
    return 0;

  } catch (const c10::Error& e) {
    std::cerr << "Error: " << e.msg() << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
