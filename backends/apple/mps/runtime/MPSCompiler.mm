//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

// Obj-C headers
#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

// MPS headers
#include <executorch/backends/apple/mps/runtime/MPSDevice.h>
#include <executorch/backends/apple/mps/runtime/MPSCompiler.h>
#include <executorch/backends/apple/mps/runtime/MPSGraphBuilder.h>
#include <executorch/backends/apple/mps/schema_generated.h>

// Runtime headers
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include <unordered_map>
#include <string>
#include <iostream>

#define CAPTURE_MODEL 1
#define MPS_UNUSED(x) ( (void)(x) )

namespace torch {
namespace executor {
namespace mps {
namespace delegate {

void printLoadedGraph(MPSGraphExecutable* executable) {
  NSLog(@"Loaded graph: %@", [executable debugDescription]);
}

MPSGraphExecutable* loadExecutable() {
  NSError* error = nil;
  // NSString* packageName = [NSString stringWithUTF8String:(
      // std::string("%@/mpsgraphmodule_") + std::to_string(arc4random_uniform(INT_MAX)) + ".mpsgraphpackage").c_str()];

  NSString* packageName = [NSString stringWithUTF8String:std::string("%@/mpsgraphmodule.mpsgraphpackage").c_str()];
#if TARGET_OS_IPHONE
  NSArray *paths = NSSearchPathForDirectoriesInDomains
      (NSDocumentDirectory, NSUserDomainMask, YES);
  NSString *documentsDirectory = [paths objectAtIndex:0];
#else
  NSString *documentsDirectory = @"/tmp";
#endif

  NSLog(@"Dcouments directory: %@", documentsDirectory);
  NSString *dataFileNSStr = [NSString stringWithFormat:packageName,
                                                        documentsDirectory];
  NSLog(@"File path: %@", dataFileNSStr);

  // NSString* manifestFileStr = [NSString stringWithFormat:@"%@/manifest.plist", dataFileNSStr];
  // NSString* model0FileStr = [NSString stringWithFormat:@"%@/model_0.mpsgraph", dataFileNSStr];
  // NSString* model1FileStr = [NSString stringWithFormat:@"%@/model_1.mpsgraph", dataFileNSStr];

  // NSFileManager *fileManager= [NSFileManager defaultManager];
  // [fileManager createDirectoryAtPath:dataFileNSStr withIntermediateDirectories:NO attributes:nil error:&error];

  // [new_manifest_plist_data writeToFile:manifestFileStr options:NSDataWritingAtomic error:&error];
  // [new_model_0_data writeToFile:model0FileStr options:NSDataWritingAtomic error:&error];
  // [new_model_1_data writeToFile:model1FileStr options:NSDataWritingAtomic error:&error];

  NSURL *bundleURL = [NSURL fileURLWithPath:dataFileNSStr];
  MPSGraphCompilationDescriptor *compilationDescriptor = [MPSGraphCompilationDescriptor new];
  compilationDescriptor.optimizationLevel = MPSGraphOptimizationLevel0;
  MPSGraphExecutable *newExec = [[MPSGraphExecutable new] initWithMPSGraphPackageAtURL:bundleURL compilationDescriptor:compilationDescriptor];

  // assert(newExec != nil);
  // [newExec dump];
  return newExec;
}
/*
Builds the mps runtime object using the buffer pointer. The buffer pointer
must be a valid pointer to the serialized mps object.
*/
__ET_NODISCARD Error MPSCompiler::compileModel(
  const void* buffer_pointer,
  size_t num_bytes,
  MPSExecutor* executor,
  MemoryAllocator* runtime_allocator,
  ArrayRef<CompileSpec> compile_specs) {
  MPS_UNUSED(compile_specs);

  Error err = Error::Ok;




#if CAPTURE_MODEL
/**
  std::unique_ptr<MPSGraphBuilder> mpsGraphBuilder(
    new MPSGraphBuilder(buffer_pointer, executor->_mpsGraphTensorToId));
  err = mpsGraphBuilder->compileModel();
  ET_CHECK_OR_RETURN_ERROR(
    err == Error::Ok, Internal, "Failed to construct the MPS graph object");

  executor->_executable = mpsGraphBuilder->getMPSGraphExecutable();
  ET_CHECK_OR_RETURN_ERROR(
      executor->_executable != nil,
      InvalidProgram,
      "Invalid FlatBuffer contents - could not create MPSGraphExecutable");
*/
  std::cout << ">> Writing executable to /tmp folder!!\n";
  MPSGraphExecutableSerializationDescriptor *serializationDescriptor = [MPSGraphExecutableSerializationDescriptor new];
  std::string dataFolder = "/tmp/";

  std::string name = "mpsgraphmodule";
  std::string mpsgraphpackagePath = dataFolder + name + ".mpsgraphpackage";

  NSString *mpsgraphpackageFileStr = [NSString stringWithUTF8String:mpsgraphpackagePath.c_str()];
  NSURL *bundleURL = [NSURL fileURLWithPath:mpsgraphpackageFileStr];

  serializationDescriptor.deploymentPlatform = MPSGraphDeploymentPlatformIOS;
  serializationDescriptor.minimumDeploymentTarget = @"1.0.0";

  [executor->_executable serializeToMPSGraphPackageAtURL:bundleURL descriptor:serializationDescriptor];
#else
  executor->_executable = loadExecutable();
#endif
  err = executor->initDataBuffers();
  ET_CHECK_OR_RETURN_ERROR(
      err == Error::Ok, Internal, "Could not allocate data buffers");

  ET_LOG(Debug, "MPSGraphExecutable total inputs: %lu", [executor->_inputShapes count]);
  ET_LOG(Debug, "MPSGraphExecutable total outputs: %lu", [executor->_outputShapes count]);

  // [executor->_executable specializeWithDevice:[MPSGraphDevice deviceWithMTLDevice:MTLCreateSystemDefaultDevice()]
  //                 inputTypes:executor->_inputShapes
  //       compilationDescriptor:nil];

  return err;
}

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
