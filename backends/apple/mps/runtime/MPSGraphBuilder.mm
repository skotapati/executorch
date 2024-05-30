//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include <executorch/backends/apple/mps/runtime/MPSGraphBuilder.h>
#include <executorch/backends/apple/mps/runtime/MPSDevice.h>
#include <iostream>


@interface MPSGraph()
-(void)dump;
@end

namespace torch {
namespace executor {
namespace mps {
namespace delegate {

MPSGraphBuilder::MPSGraphBuilder(const void* buffer_pointer, std::unordered_map<MPSGraphTensor*, int32_t>& mpsGraphTensorToId) : _mpsGraphTensorToId(mpsGraphTensorToId), _buffer_pointer(buffer_pointer) {
  _mpsGraph = [MPSGraph new];
  _feeds = [NSMutableDictionary dictionary];
  _targetTensors = [NSMutableArray new];

  _mpsGraphExecutable = nil;
  _metal_kernel = false;
}

Error
MPSGraphBuilder::compileModel() {
  Error err = Error::Ok;

  // uint8_t* flatbuffer_start = (uint8_t*)_buffer_pointer + 4 + 4 + 8 + 8;
  uint8_t* flatbuffer_start = (uint8_t*)_buffer_pointer;
  uint32_t start = *((uint32_t*)flatbuffer_start);
  flatbuffer_start = flatbuffer_start + 4;
  ET_CHECK(start == 0);
  ET_CHECK((char)flatbuffer_start[0] == 'M');
  ET_CHECK((char)flatbuffer_start[1] == 'P');
  ET_CHECK((char)flatbuffer_start[2] == '0');
  ET_CHECK((char)flatbuffer_start[3] == '1');
  // ET_CHECK(flatbuffers::BufferHasIdentifier(flatbuffer_start, fbIdentifier));
  flatbuffer_start = flatbuffer_start + 4;
  data_segment_offset = *((uint64_t*)flatbuffer_start);
  flatbuffer_start = flatbuffer_start + 8;
  data_segment_size = *((uint64_t*)flatbuffer_start);
  flatbuffer_start = flatbuffer_start + 8;
  std::cout << "data_segment_offset = " << data_segment_offset << std::endl;
  std::cout << "data_segment_size = " << data_segment_size << std::endl;

  constant_data_ptr = (uint8_t*)_buffer_pointer + data_segment_offset;

  ET_CHECK(flatbuffer_start != nullptr);
  ET_CHECK_OR_RETURN_ERROR(
    mpsgraph::MPSGraphBufferHasIdentifier(flatbuffer_start),
    DelegateInvalidCompatibility,
    "MPS Delegate Serialization Format version identifier '%.4s' != expected '%.4s'",
    flatbuffers::GetBufferIdentifier(flatbuffer_start),
    mpsgraph::MPSGraphIdentifier());

  _flatBufferGraph = mpsgraph::GetMPSGraph(flatbuffer_start);
  switch (_flatBufferGraph->graph_type()) {
    case mpsgraph::OpType::metal_kernel:
    {
      _metal_kernel = true;
      err = compileMetalKernel();
      break;
    }
    case mpsgraph::OpType::mps_graph:
    {
      err = compileMPSGraph();
      break;
    }
    default:
      ET_CHECK_OR_RETURN_ERROR(
      false,
      DelegateInvalidCompatibility,
      "Received an invalid operation type: expected MPSGraph or metal kernel, but got: %s",
      EnumNameOpType(_flatBufferGraph->graph_type()));
  }

  return err;
}

Error
MPSGraphBuilder::compileMPSGraph() {
  Error err = Error::Ok;

  _idToMPSGraphTensor.resize(_flatBufferGraph->mps_values()->size(), nullptr);
  // std::cout << "Constant segment size: " <<   _flatBufferGraph->constant_segment()->size() << std::endl;
  // std::cout << "Constant segment offset: " <<   _flatBufferGraph->constant_segment()->offset() << std::endl;
  std::cout << "sizeof: " << sizeof(*_flatBufferGraph) << " " << sizeof(_flatBufferGraph) << " " <<  std::endl;
  // Add the placeholder nodes to the graph.
  for (auto in_id : *_flatBufferGraph->input_ids()) {
    err = mpsGraphRankedPlaceholder(in_id);
    if (err != Error::Ok) {
      return err;
    }
  }

  // Parse all the serialized constant values and add them to MPSGraph.
  for (auto constant_id : *_flatBufferGraph->constant_ids()) {
    err = mpsConstantOp(constant_id);
    if (err != Error::Ok) {
      return err;
    }
  }

  // Create the corresponding MPSGraph ops of the serialized nodes from the FlatBuffer.
  for (auto node : *_flatBufferGraph->mps_nodes()) {
    err = addNodeToMPSGraph(node);
    if (err != Error::Ok) {
      return err;
    }
  }

  // Add the output nodes to the MPSGraphExecutable.
  for (auto out_id : *_flatBufferGraph->output_ids()) {
    ET_CHECK_OR_RETURN_ERROR(
      _idToMPSGraphTensor[out_id] != nil,
      InvalidState,
      "Failed to deserialize the model");

    [_targetTensors addObject: _idToMPSGraphTensor[out_id]];
  }

  [_mpsGraph dump];

  return err;
}

Error
MPSGraphBuilder::compileMetalKernel() {
  Error err = Error::Ok;

  ET_CHECK_OR_RETURN_ERROR(
    _flatBufferGraph->mps_nodes()->size() == 1,
    DelegateInvalidCompatibility,
    "Currently supporting dispatching a single Metal kernel.");
  ET_CHECK_OR_RETURN_ERROR(
    _flatBufferGraph->constant_ids()->size() == 0,
    DelegateInvalidCompatibility,
    "Currently not supporting dispatching Metal kernels with constants.");

  // Compile the corresponding Metal kernel
  for (auto node : *_flatBufferGraph->mps_nodes()) {
    err = compileMetalKernel(node);
    if (err != Error::Ok) {
      return err;
    }
  }

  return err;
}

Error
MPSGraphBuilder::mpsGraphRankedPlaceholder(int32_t id) {
  ET_LOG(Debug, "%s: %d", __FUNCTION__, id);
  MPSShape* mpsShape = getMPSShape(id);
  MPSDataType mpsDataType = getMPSDataType(id);
  MPSGraphTensor* placeholder = [_mpsGraph placeholderWithShape:mpsShape
                                                  dataType:mpsDataType
                                                      name:nil];
  _idToMPSGraphTensor[id] = placeholder;
  _feeds[placeholder] = [[MPSGraphShapedType alloc] initWithShape:mpsShape
                                                         dataType:mpsDataType];
  _mpsGraphTensorToId[placeholder] = id;
  return Error::Ok;
}

MPSGraph*
MPSGraphBuilder::getMPSGraph() {
  return _mpsGraph;
}

MPSGraphExecutable*
MPSGraphBuilder::getMPSGraphExecutable() {
  if (_mpsGraphExecutable) {
    return _mpsGraphExecutable;
  }
  _mpsGraphExecutable = [_mpsGraph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:MPSDevice::getInstance()->device()]
                                               feeds:_feeds
                                       targetTensors:_targetTensors
                                    targetOperations:nil
                               compilationDescriptor:nil];

  return _mpsGraphExecutable;

}

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
