
//
//  Copyright (c) 2024 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include <executorch/backends/apple/mps/runtime/MPSGraphBuilder.h>

namespace torch {
namespace executor {
namespace mps {
namespace delegate {

Error
MPSGraphBuilder::mpsDequantizePerChannelGroupOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSDequantizePerChannelGroup();
  ET_LOG(
    Debug, "%s: (%d, %d, %d) -> %d",
    __FUNCTION__,
    graphNode->input1_id(),
    graphNode->scales_id(),
    graphNode->zero_points_id(),
    graphNode->output_id()
  );

  MPSGraphTensor* inputTensor = getMPSGraphTensor(graphNode->input1_id());
  MPSGraphTensor* scalesTensor = getMPSGraphTensor(graphNode->scales_id());

//   MPSGraphTensor* indexTensor = getMPSGraphTensor(graphNode->index_id());
//   MPSGraphTensor* castIndexTensor = indexTensor;
//   if(castIndexTensor.dataType != MPSDataTypeInt32) {
//     castIndexTensor = [_mpsGraph castTensor:indexTensor
//                                      toType:MPSDataTypeInt32
//                                        name:@"castTensor"];
//   }

  //     MPSGraphTensor *wDqTensor = [mpsGraph dequantizeTensor:newCachedGraph->BTensor
  //                                                scaleTensor:newCachedGraph->scalesTensor
  //                                            zeroPointTensor:zpTensor
  //                                                  minTensor:newCachedGraph->minTensor
  //                                                   dataType:getMPSScalarType(A)
  //                                                       name:nil];

  MPSGraphTensor *zpTensor = [_mpsGraph constantWithScalar:0
                                                  dataType:MPSDataTypeInt4];

  MPSGraphTensor *minTensor = [_mpsGraph constantWithScalar:0
                                                  dataType:MPSDataTypeFloat16];

  MPSGraphTensor *wDqTensor = [_mpsGraph dequantizeTensor:inputTensor
                                              scaleTensor:scalesTensor
                                          zeroPointTensor:zpTensor
                                                minTensor:minTensor
                                                dataType:MPSDataTypeFloat16
                                                    name:nil];

  _idToMPSGraphTensor[graphNode->output_id()] = wDqTensor;
  return Error::Ok;
}



} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
