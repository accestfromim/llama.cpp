#import "ggml-metal-device.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#import "ggml-impl.h"
#import "ggml-threading.h"

#include <Foundation/Foundation.h>

#include <Metal/Metal.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#ifndef TARGET_OS_VISION
#define TARGET_OS_VISION 0
#endif

// create residency sets only on macOS >= 15.0
#if !TARGET_CPU_X86_64 && TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000 || \
    TARGET_OS_IOS && __IPHONE_OS_VERSION_MAX_ALLOWED >= 180000 || \
    TARGET_OS_TV && __TV_OS_VERSION_MAX_ALLOWED >= 180000 || \
    TARGET_OS_VISION && __VISION_OS_VERSION_MAX_ALLOWED >= 200000
#define GGML_METAL_HAS_RESIDENCY_SETS 1
#endif

// overload of MTLGPUFamilyMetal3 (not available in some environments)
static const NSInteger MTLGPUFamilyMetal3_GGML = 5001;
static const NSInteger MTLGPUFamilyMetal4_GGML = 5002;

static bool ggml_metal_env_flag(const char * name, bool default_value) {
    const char * value = getenv(name);
    if (value == NULL) {
        return default_value;
    }
    return strcmp(value, "0") != 0 && strcasecmp(value, "false") != 0 && strcasecmp(value, "off") != 0;
}

#if !GGML_METAL_EMBED_LIBRARY
// Here to assist with NSBundle Path Hack
@interface GGMLMetalClass : NSObject
@end
@implementation GGMLMetalClass
@end
#endif

//
// MTLFunctionConstantValues wrapper
//

struct ggml_metal_cv {
    MTLFunctionConstantValues * obj;
};

ggml_metal_cv_t ggml_metal_cv_init(void) {
    ggml_metal_cv_t res = calloc(1, sizeof(struct ggml_metal_cv));

    res->obj = [[MTLFunctionConstantValues alloc] init];

    return res;
}

void ggml_metal_cv_free(ggml_metal_cv_t cv) {
    [cv->obj release];
    free(cv);
}

void ggml_metal_cv_set_int16(ggml_metal_cv_t cv, int16_t value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeShort atIndex:idx];
}

void ggml_metal_cv_set_int32(ggml_metal_cv_t cv, int32_t value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeInt atIndex:idx];
}

void ggml_metal_cv_set_bool(ggml_metal_cv_t cv, bool value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeBool atIndex:idx];
}

//
// MTLComputePipelineState wrapper
//

struct ggml_metal_pipeline {
    id<MTLComputePipelineState> obj;

    // suggested dispatch sizes
    int nsg;

    int nr0;
    int nr1;

    size_t smem;
};

ggml_metal_pipeline_t ggml_metal_pipeline_init(void) {
    ggml_metal_pipeline_t res = calloc(1, sizeof(struct ggml_metal_pipeline));

    *res = (struct ggml_metal_pipeline) {
        /*.obj  =*/ nil,
        /*.nsg  =*/ 0,
        /*.nr0  =*/ 0,
        /*.nr1  =*/ 0,
        /*.smem =*/ 0,
    };

    return res;
}

void ggml_metal_pipeline_free(ggml_metal_pipeline_t pipeline) {
    [pipeline->obj release];

    free(pipeline);
}

void ggml_metal_pipeline_set_nsg(ggml_metal_pipeline_t pipeline, int nsg) {
    pipeline->nsg = nsg;
}

int ggml_metal_pipeline_get_nsg(ggml_metal_pipeline_t pipeline) {
    return pipeline->nsg;
}

void ggml_metal_pipeline_set_nr0(ggml_metal_pipeline_t pipeline, int nr0) {
    pipeline->nr0 = nr0;
}

int ggml_metal_pipeline_get_nr0(ggml_metal_pipeline_t pipeline) {
    return pipeline->nr0;
}

void ggml_metal_pipeline_set_nr1(ggml_metal_pipeline_t pipeline, int nr1) {
    pipeline->nr1 = nr1;
}

int ggml_metal_pipeline_get_nr1(ggml_metal_pipeline_t pipeline) {
    return pipeline->nr1;
}

void   ggml_metal_pipeline_set_smem(ggml_metal_pipeline_t pipeline, size_t smem) {
    pipeline->smem = smem;
}

size_t ggml_metal_pipeline_get_smem(ggml_metal_pipeline_t pipeline) {
    return pipeline->smem;
}

int ggml_metal_pipeline_max_theads_per_threadgroup(ggml_metal_pipeline_t pipeline) {
    return pipeline->obj.maxTotalThreadsPerThreadgroup;
}

struct ggml_metal_library {
    id<MTLLibrary> obj;
    id<MTLDevice> device;

    ggml_metal_pipelines_t pipelines; // cache of compiled pipelines
};

ggml_metal_library_t ggml_metal_library_init(ggml_metal_device_t dev) {
    id<MTLLibrary> library = nil;
    id<MTLDevice> device = ggml_metal_device_get_obj(dev);

    // load library
    //
    // - first check if the library is embedded
    // - then check if the library is in the bundle
    // - if not found, load the source and compile it
    // - if that fails, return NULL
    //
    // TODO: move to a function
    {
        const int64_t t_start = ggml_time_us();

        NSError * error = nil;
        NSString * src = nil;

#if GGML_METAL_EMBED_LIBRARY
        GGML_LOG_INFO("%s: using embedded metal library\n", __func__);

        extern const char ggml_metallib_start[];
        extern const char ggml_metallib_end[];

        src = [[NSString alloc] initWithBytes:ggml_metallib_start length:(ggml_metallib_end-ggml_metallib_start) encoding:NSUTF8StringEncoding];
#else

#ifdef SWIFT_PACKAGE
        NSBundle * bundle = SWIFTPM_MODULE_BUNDLE;
#else
        NSBundle * bundle = [NSBundle bundleForClass:[GGMLMetalClass class]];
#endif

        const bool force_source = getenv("GGML_METAL_FORCE_SOURCE") != NULL;
        NSString * path_lib     = force_source ? nil : [bundle pathForResource:@"default" ofType:@"metallib"];
        if (path_lib == nil && !force_source) {
            // Try to find the resource in the directory where the current binary located.
            NSString * bin_cur = [[NSProcessInfo processInfo] arguments][0];
            NSString * bin_dir = [bin_cur stringByDeletingLastPathComponent];

            NSString * path_lib_default = [NSString pathWithComponents:@[bin_dir, @"default.metallib"]];
            if ([[NSFileManager defaultManager] isReadableFileAtPath:path_lib_default]) {
                GGML_LOG_INFO("%s: found '%s'\n", __func__, [path_lib_default UTF8String]);

                NSDictionary * atts = [[NSFileManager defaultManager] attributesOfItemAtPath:path_lib_default error:&error];
                if (atts && atts[NSFileType] == NSFileTypeSymbolicLink) {
                    // Optionally, if this is a symlink, try to resolve it.
                    path_lib_default = [[NSFileManager defaultManager] destinationOfSymbolicLinkAtPath:path_lib_default error:&error];
                    if (path_lib_default && [path_lib_default length] > 0 && ![[path_lib_default substringToIndex:1] isEqualToString:@"/"]) {
                        // It is a relative path, adding the binary directory as directory prefix.
                        path_lib_default = [NSString pathWithComponents:@[bin_dir, path_lib_default]];
                    }
                    if (!path_lib_default || ![[NSFileManager defaultManager] isReadableFileAtPath:path_lib_default]) {
                        // Link to the resource could not be resolved.
                        path_lib_default = nil;
                    } else {
                        GGML_LOG_INFO("%s: symlink resolved '%s'\n", __func__, [path_lib_default UTF8String]);
                    }
                }
            } else {
                // The resource couldn't be found in the binary's directory.
                path_lib_default = nil;
            }

            path_lib = path_lib_default;
        }

        if (path_lib != nil) {
            // pre-compiled library found
            NSURL * libURL = [NSURL fileURLWithPath:path_lib];
            GGML_LOG_INFO("%s: loading '%s'\n", __func__, [path_lib UTF8String]);

            library = [device newLibraryWithURL:libURL error:&error];
            if (error) {
                GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                return nil;
            }
        } else {
            GGML_LOG_INFO("%s: default.metallib not found, loading from source\n", __func__);

            NSString * path_source;
            NSString * path_resource = [[NSProcessInfo processInfo].environment objectForKey:@"GGML_METAL_PATH_RESOURCES"];

            GGML_LOG_INFO("%s: GGML_METAL_PATH_RESOURCES = %s\n", __func__, path_resource ? [path_resource UTF8String] : "nil");

            if (path_resource) {
                path_source = [path_resource stringByAppendingPathComponent:@"ggml-metal.metal"];
            } else {
                path_source = [bundle pathForResource:@"ggml-metal" ofType:@"metal"];
            }

            if (path_source == nil) {
                GGML_LOG_WARN("%s: error: could not use bundle path to find ggml-metal.metal, falling back to trying cwd\n", __func__);
                path_source = @"ggml-metal.metal";
            }

            GGML_LOG_INFO("%s: loading '%s'\n", __func__, [path_source UTF8String]);

            src = [NSString stringWithContentsOfFile:path_source encoding:NSUTF8StringEncoding error:&error];
            if (error) {
                GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                return nil;
            }
        }
#endif

        if (!library) {
            @autoreleasepool {
                // dictionary of preprocessor macros
                NSMutableDictionary * prep = [NSMutableDictionary dictionary];

                if (ggml_metal_device_get_props(dev)->has_bfloat) {
                    [prep setObject:@"1" forKey:@"GGML_METAL_HAS_BF16"];
                }

#if GGML_METAL_EMBED_LIBRARY
                [prep setObject:@"1" forKey:@"GGML_METAL_EMBED_LIBRARY"];
#endif

                MTLCompileOptions * options = [MTLCompileOptions new];
                options.preprocessorMacros = prep;

                //[options setFastMathEnabled:false];

                library = [device newLibraryWithSource:src options:options error:&error];
                if (error) {
                    GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                    return nil;
                }

#if !__has_feature(objc_arc)
                [options release];
#endif
            }
        }

#if GGML_METAL_EMBED_LIBRARY
        [src release];
#endif // GGML_METAL_EMBED_LIBRARY

        GGML_LOG_INFO("%s: loaded in %.3f sec\n", __func__, (ggml_time_us() - t_start) / 1e6);
    }

    ggml_metal_library_t res = calloc(1, sizeof(struct ggml_metal_library));

    res->obj = library;
    res->device = device;
    res->pipelines = ggml_metal_pipelines_init();

    return res;
}

void ggml_metal_library_free(ggml_metal_library_t lib) {
    if (!lib) {
        return;
    }

    if (lib->obj) {
        [lib->obj release];
    }

    ggml_metal_pipelines_free(lib->pipelines);

    free(lib);
}

ggml_metal_pipeline_t ggml_metal_library_get_pipeline(ggml_metal_library_t lib, const char * name) {
    return ggml_metal_pipelines_get(lib->pipelines, name);
}

ggml_metal_pipeline_t ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, const char * base, const char * name, ggml_metal_cv_t cv) {
    // note: the pipelines are cached in the library per device, so they are shared across all metal contexts
    ggml_critical_section_start();

    ggml_metal_pipeline_t res = ggml_metal_library_get_pipeline(lib, name);
    if (res) {
        ggml_critical_section_end();

        return res;
    }

    res = ggml_metal_pipeline_init();

    @autoreleasepool {
        NSError * error = nil;

        NSString * base_func = [NSString stringWithUTF8String:base];

        GGML_LOG_DEBUG("%s: compiling pipeline: base = '%s', name = '%s'\n", __func__, base, name);

        id<MTLFunction> mtl_function;
        if (!cv) {
            mtl_function = [lib->obj newFunctionWithName:base_func];
        } else {
            mtl_function = [lib->obj newFunctionWithName:base_func constantValues:cv->obj error:&error];
        }
        if (!mtl_function) {
            ggml_critical_section_end();

            GGML_LOG_ERROR("%s: error: failed to compile pipeline: base = '%s', name = '%s'\n", __func__, base, name);
            if (error) {
                GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
            }

            return nil;
        }

        res->obj = [lib->device newComputePipelineStateWithFunction:mtl_function error:&error];

        ggml_metal_pipelines_add(lib->pipelines, name, res);

        [mtl_function release];

        GGML_LOG_DEBUG("%s: loaded %-40s %16p | th_max = %4d | th_width = %4d\n", __func__, name, (void *) res->obj,
                (int) res->obj.maxTotalThreadsPerThreadgroup,
                (int) res->obj.threadExecutionWidth);
    }

    ggml_critical_section_end();

    return res;
}

//
// MTLComputeCommandEncoder wrapper
//

struct ggml_metal_encoder {
    id<MTLComputeCommandEncoder> obj;
};

ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent) {
    ggml_metal_encoder_t res = calloc(1, sizeof(struct ggml_metal_encoder));

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    if (concurrent) {
        res->obj = [cmd_buf computeCommandEncoderWithDispatchType: MTLDispatchTypeConcurrent];
    } else {
        res->obj = [cmd_buf computeCommandEncoder];
    }

    [res->obj retain];

    return res;
}

void ggml_metal_encoder_free(ggml_metal_encoder_t encoder) {
    [encoder->obj release];
    free(encoder);
}

void ggml_metal_encoder_debug_group_push(ggml_metal_encoder_t encoder, const char * name) {
    [encoder->obj pushDebugGroup:[NSString stringWithCString:name encoding:NSUTF8StringEncoding]];
}

void ggml_metal_encoder_debug_group_pop (ggml_metal_encoder_t encoder) {
    [encoder->obj popDebugGroup];
}

void ggml_metal_encoder_set_pipeline(ggml_metal_encoder_t encoder, ggml_metal_pipeline_t pipeline) {
    [encoder->obj setComputePipelineState:pipeline->obj];
}

void ggml_metal_encoder_set_bytes(ggml_metal_encoder_t encoder, void * data, size_t size, int idx) {
    [encoder->obj setBytes:data length:size atIndex:idx];
}

void ggml_metal_encoder_set_buffer(ggml_metal_encoder_t encoder, struct ggml_metal_buffer_id buffer, int idx) {
    [encoder->obj setBuffer:buffer.metal offset:buffer.offs atIndex:idx];
}

void ggml_metal_encoder_set_threadgroup_memory_size(ggml_metal_encoder_t encoder, size_t size, int idx) {
    [encoder->obj setThreadgroupMemoryLength:size atIndex:idx];
}

void ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, int tg0, int tg1, int tg2, int tptg0, int tptg1, int tptg2) {
    [encoder->obj dispatchThreadgroups:MTLSizeMake(tg0, tg1, tg2) threadsPerThreadgroup:MTLSizeMake(tptg0, tptg1, tptg2)];
}

void ggml_metal_encoder_memory_barrier(ggml_metal_encoder_t encoder) {
    [encoder->obj memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void ggml_metal_encoder_end_encoding(ggml_metal_encoder_t encoder) {
    [encoder->obj endEncoding];
}

struct ggml_metal_device {
    id<MTLDevice> mtl_device;

    // a single global queue shared by all Metal backends
    // technically not needed for devices with unified memory, but enables discrete GPUs support
    // ref: https://github.com/ggml-org/llama.cpp/pull/15906
    id<MTLCommandQueue> mtl_queue;

    ggml_metal_library_t library;

    struct ggml_metal_device_props props;

    ggml_metal_buffer_t scratch;
    size_t scratch_size;
};

ggml_metal_device_t ggml_metal_device_init(void) {
    ggml_metal_device_t dev = calloc(1, sizeof(struct ggml_metal_device));

    assert(dev != NULL);

    if (dev->mtl_device == nil) {
        dev->mtl_device = MTLCreateSystemDefaultDevice();

        if (dev->mtl_device == nil) {
            GGML_LOG_ERROR("%s: error: no Metal device found (disabling Metal backend)\n", __func__);
            free(dev);
            return NULL;
        }

        dev->mtl_queue = [dev->mtl_device newCommandQueue];
        if (dev->mtl_queue == nil) {
            GGML_LOG_ERROR("%s: error: failed to create command queue (disabling Metal backend)\n", __func__);
            [dev->mtl_device release];
            dev->mtl_device = nil;
            free(dev);
            return NULL;
        }

            dev->props.has_simdgroup_reduction  = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];
            dev->props.has_simdgroup_reduction |= [dev->mtl_device supportsFamily:MTLGPUFamilyMetal3_GGML];

            dev->props.has_simdgroup_mm = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];
            dev->props.has_unified_memory = dev->mtl_device.hasUnifiedMemory;

            dev->props.has_bfloat  = [dev->mtl_device supportsFamily:MTLGPUFamilyMetal3_GGML];
            dev->props.has_bfloat |= [dev->mtl_device supportsFamily:MTLGPUFamilyApple6];
#if !defined(GGML_METAL_EMBED_LIBRARY) && !defined(GGML_METAL_HAS_BF16_LIBRARY)
            // The precompiled library was built without its BF16 entry points.
            dev->props.has_bfloat = false;
#endif

            dev->props.use_residency_sets = true;
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
            dev->props.use_residency_sets = getenv("GGML_METAL_NO_RESIDENCY") == nil;
#endif

            dev->props.use_shared_buffers = dev->props.has_unified_memory;

            if (getenv("GGML_METAL_SHARED_BUFFERS_DISABLE") != NULL) {
                dev->props.use_shared_buffers = false;
            }

            const bool supports_metal4 = [dev->mtl_device supportsFamily:MTLGPUFamilyMetal4_GGML];
            dev->props.fairy2i_metal3_compat = ggml_metal_env_flag(
                "GGML_FAIRY2I_METAL3_COMPAT", !supports_metal4);

            dev->props.supports_gpu_family_apple7 = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];

            dev->props.max_buffer_size            = dev->mtl_device.maxBufferLength;
            dev->props.max_working_set_size       = dev->mtl_device.recommendedMaxWorkingSetSize;
            dev->props.max_theadgroup_memory_size = dev->mtl_device.maxThreadgroupMemoryLength;

            strncpy(dev->props.name, [[dev->mtl_device name] UTF8String], sizeof(dev->props.name) - 1);

            dev->library = ggml_metal_library_init(dev);
            if (!dev->library) {
                GGML_LOG_ERROR("%s: error: failed to create library (disabling Metal backend)\n", __func__);
                ggml_metal_device_free(dev);
                return NULL;
            }

            // --------------------------------------------------

            // print MTL GPU family:
            GGML_LOG_INFO("%s: GPU name:   %s\n", __func__, dev->props.name);

            // determine max supported GPU family
            // https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
            // https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
            {
                for (int i = MTLGPUFamilyApple1 + 20; i >= MTLGPUFamilyApple1; --i) {
                    if ([dev->mtl_device supportsFamily:i]) {
                        GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyApple%d  (%d)\n", __func__, i - (int) MTLGPUFamilyApple1 + 1, i);
                        break;
                    }
                }

                for (int i = MTLGPUFamilyCommon1 + 5; i >= MTLGPUFamilyCommon1; --i) {
                    if ([dev->mtl_device supportsFamily:i]) {
                        GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyCommon%d (%d)\n", __func__, i - (int) MTLGPUFamilyCommon1 + 1, i);
                        break;
                    }
                }

                for (int i = MTLGPUFamilyMetal3_GGML + 5; i >= MTLGPUFamilyMetal3_GGML; --i) {
                    if ([dev->mtl_device supportsFamily:i]) {
                        GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyMetal%d  (%d)\n", __func__, i - (int) MTLGPUFamilyMetal3_GGML + 3, i);
                        break;
                    }
                }
            }

            GGML_LOG_INFO("%s: simdgroup reduction   = %s\n", __func__, dev->props.has_simdgroup_reduction ? "true" : "false");
            GGML_LOG_INFO("%s: simdgroup matrix mul. = %s\n", __func__, dev->props.has_simdgroup_mm        ? "true" : "false");
            GGML_LOG_INFO("%s: has unified memory    = %s\n", __func__, dev->props.has_unified_memory      ? "true" : "false");
            GGML_LOG_INFO("%s: has bfloat            = %s\n", __func__, dev->props.has_bfloat              ? "true" : "false");
            GGML_LOG_INFO("%s: use residency sets    = %s\n", __func__, dev->props.use_residency_sets      ? "true" : "false");
            GGML_LOG_INFO("%s: use shared buffers    = %s\n", __func__, dev->props.use_shared_buffers      ? "true" : "false");
            GGML_LOG_INFO("%s: Fairy2i Metal3 compat = %s\n", __func__, dev->props.fairy2i_metal3_compat   ? "true" : "false");

#if TARGET_OS_OSX || (TARGET_OS_IOS && __clang_major__ >= 15)
            if (@available(macOS 10.12, iOS 16.0, *)) {
                GGML_LOG_INFO("%s: recommendedMaxWorkingSetSize  = %8.2f MB\n", __func__, dev->props.max_working_set_size / 1e6);
            }
#endif
    }

    return dev;
}

void ggml_metal_device_free(ggml_metal_device_t dev) {
    assert(dev != NULL);

    if (dev->scratch) {
        ggml_metal_buffer_free(dev->scratch);
        dev->scratch = NULL;
        dev->scratch_size = 0;
    }

    ggml_metal_library_free(dev->library);
    dev->library = NULL;

    if (dev->mtl_queue) {
        [dev->mtl_queue release];
        dev->mtl_queue = nil;
    }

    if (dev->mtl_device) {
        [dev->mtl_device release];
        dev->mtl_device = nil;
    }

    free(dev);
}

void * ggml_metal_device_get_obj(ggml_metal_device_t dev) {
    return dev->mtl_device;
}

void * ggml_metal_device_get_queue(ggml_metal_device_t dev) {
    return dev->mtl_queue;
}

ggml_metal_library_t ggml_metal_device_get_library(ggml_metal_device_t dev) {
    return dev->library;
}

void ggml_metal_device_get_memory(ggml_metal_device_t dev, size_t * free, size_t * total) {
    if (@available(macOS 10.12, iOS 16.0, *)) {
        *total = dev->mtl_device.recommendedMaxWorkingSetSize;
        *free  = *total - dev->mtl_device.currentAllocatedSize;
    } else {
        *free = 0;
        *total = 0;
    }
}

bool ggml_metal_device_supports_op(ggml_metal_device_t dev, const struct ggml_tensor * op) {
    const bool has_simdgroup_mm        = dev->props.has_simdgroup_mm;
    const bool has_simdgroup_reduction = dev->props.has_simdgroup_reduction;
    const bool has_bfloat              = dev->props.has_bfloat;

    if (op->op == GGML_OP_ROW4_LINEAR || op->op == GGML_OP_W8A8_LINEAR) {
        const struct ggml_tensor * x      = op->src[0];
        const struct ggml_tensor * codes  = op->src[1];
        const struct ggml_tensor * scales = op->src[2];
        const int32_t              layout = ggml_get_op_params_i32(op, 0);
        const int32_t              m      = ggml_get_op_params_i32(op, 1);
        const int32_t              k      = ggml_get_op_params_i32(op, 2);

        if (!has_simdgroup_mm || !has_simdgroup_reduction || !x || !codes || !scales || op->src[3] ||
            m <= 0 || k <= 0 || k % 128 != 0 || op->type != GGML_TYPE_F32 ||
            x->type != GGML_TYPE_F32 || x->ne[0] != k || op->ne[0] != m || op->ne[1] != x->ne[1] ||
            op->ne[2] != x->ne[2] || op->ne[3] != x->ne[3] || scales->ne[0] != m || scales->ne[1] != 1 ||
            scales->ne[2] != 1 || scales->ne[3] != 1 || !ggml_is_contiguous(x) || !ggml_is_contiguous(codes) ||
            !ggml_is_contiguous(scales) || !ggml_is_contiguous(op)) {
            return false;
        }

        if (op->op == GGML_OP_ROW4_LINEAR) {
            // Pair2 B1 stages the full-K activation in threadgroup memory and
            // accumulates LUT-decoded integer values in F32. Every integer is
            // exactly representable through K=65536 (worst case 2^24).
            const bool pair2_decode_ok = ggml_nrows(x) != 1 ||
                                         (k <= 65536 &&
                                          (size_t) k * sizeof(int8_t) <= dev->props.max_theadgroup_memory_size);
            const bool layout_v1 =
                layout == 1 && m % 128 == 0 && codes->type == GGML_TYPE_ROW4_CODES && codes->ne[0] == 64 &&
                codes->ne[1] == 4 &&
                codes->ne[2] == k / 128 && codes->ne[3] == m / 16;
            const bool layout_v2 =
                layout == 2 && m % 32 == 0 && k % 256 == 0 && codes->type == GGML_TYPE_ROW4_CODES_PAIR2 &&
                codes->ne[0] == 128 && codes->ne[1] == 8 && codes->ne[2] == k / 256 && codes->ne[3] == m / 32 &&
                pair2_decode_ok;
            return scales->type == GGML_TYPE_BF16 && (layout_v1 || layout_v2);
        }

        return layout == 1 && m % 128 == 0 && codes->type == GGML_TYPE_I8 && scales->type == GGML_TYPE_F32 &&
               codes->ne[0] == 128 && codes->ne[1] == 16 && codes->ne[2] == k / 128 &&
               codes->ne[3] == m / 16;
    }

    if ((op->op == GGML_OP_FAIRY2I_WIDE_LINEAR_W1 || op->op == GGML_OP_FAIRY2I_WIDE_LINEAR_W2) && op->src[1] &&
        op->src[1]->type == GGML_TYPE_FAIRY2I_BUNDLE_CODES) {
        const int32_t layout   = ggml_get_op_params_i32(op, 0);
        const int32_t m        = ggml_get_op_params_i32(op, 1);
        const int32_t k        = ggml_get_op_params_i32(op, 2);
        const int32_t branches = ggml_get_op_params_i32(op, 3);
        const struct ggml_tensor * x      = op->src[0];
        const struct ggml_tensor * codes  = op->src[1];
        const struct ggml_tensor * scales = op->src[2];
        const struct ggml_tensor * bias   = op->src[3];
        const int64_t tiles    = m > 0 && k > 0 ? (int64_t) (m / 64) * (k / 64) : 0;
        const int32_t expected_branches = op->op == GGML_OP_FAIRY2I_WIDE_LINEAR_W1 ? 2 : 4;
        const bool exact_bundle =
            scales && scales->type == GGML_TYPE_BF16 &&
            ((op->op == GGML_OP_FAIRY2I_WIDE_LINEAR_W1 && branches == 2) ||
             (op->op == GGML_OP_FAIRY2I_WIDE_LINEAR_W2 && branches == 4));
        const bool valid_scale_type = scales && (scales->type == GGML_TYPE_F16 || exact_bundle);
        return has_simdgroup_mm && has_simdgroup_reduction && x && scales && layout == 1 && m > 0 && k > 0 &&
               m % 64 == 0 && k % 64 == 0 && branches == expected_branches && op->type == GGML_TYPE_F32 &&
               x->type == GGML_TYPE_F32 && valid_scale_type && (!exact_bundle || has_bfloat) &&
               (!bias || bias->type == GGML_TYPE_F32) &&
               x->ne[0] == k && op->ne[0] == m && codes->ne[0] == 16 && codes->ne[1] == branches &&
               codes->ne[2] == 64 && codes->ne[3] == tiles && scales->ne[0] == 2 && scales->ne[1] == branches &&
               scales->ne[2] == tiles && scales->ne[3] == 1 && ggml_is_contiguous(x) && ggml_is_contiguous(op) &&
               ggml_is_contiguous(codes) && ggml_is_contiguous(scales);
    }

    if (op->op == GGML_OP_FAIRY2I_WIDE_LINEAR_W2 &&
        (!op->src[1] || op->src[1]->type != GGML_TYPE_FAIRY2I_BUNDLE_CODES)) {
        const struct ggml_tensor * x    = op->src[0];
        const struct ggml_tensor * u_s0 = op->src[1];
        const struct ggml_tensor * u_s1 = op->src[2];
        const struct ggml_tensor * w_s0 = op->src[3];
        const struct ggml_tensor * w_s1 = op->src[4];
        const struct ggml_tensor * bias = op->src[5];

        if (!x || !u_s0 || !w_s0 || (u_s1 == NULL) != (w_s1 == NULL)) {
            return false;
        }

        const bool has_stage1 = u_s1 != NULL;

        return op->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32 &&
               u_s0->type == GGML_TYPE_FAIRY2I_TILE64_V2 && w_s0->type == GGML_TYPE_FAIRY2I_TILE64_V2 &&
               (!has_stage1 ||
                (u_s1->type == GGML_TYPE_FAIRY2I_TILE64_V2 && w_s1->type == GGML_TYPE_FAIRY2I_TILE64_V2)) &&
               (!bias || bias->type == GGML_TYPE_F32) && ggml_is_contiguous(x) && ggml_is_contiguous(op) &&
               ggml_is_contiguous(u_s0) && ggml_is_contiguous(w_s0) &&
               (!has_stage1 || (ggml_is_contiguous(u_s1) && ggml_is_contiguous(w_s1))) &&
               x->ne[0] % ggml_blck_size(GGML_TYPE_FAIRY2I_TILE64_V2) == 0 && op->ne[0] == u_s0->ne[1] &&
               u_s0->ne[0] == x->ne[0] && w_s0->ne[0] == x->ne[0] && w_s0->ne[1] == op->ne[0] &&
               (!has_stage1 ||
                (u_s1->ne[0] == x->ne[0] && w_s1->ne[0] == x->ne[0] && u_s1->ne[1] == op->ne[0] &&
                 w_s1->ne[1] == op->ne[0])) &&
               u_s0->ne[2] == 1 && u_s0->ne[3] == 1 && w_s0->ne[2] == 1 && w_s0->ne[3] == 1 &&
               (!has_stage1 ||
                (u_s1->ne[2] == 1 && u_s1->ne[3] == 1 && w_s1->ne[2] == 1 && w_s1->ne[3] == 1));
    }

    // Pair2 is opaque but permits an exact same-type byte copy.  No numeric
    // conversion is implied by this path.
    if (op->op == GGML_OP_CPY && op->src[0] &&
        op->src[0]->type == GGML_TYPE_ROW4_CODES_PAIR2 && op->type == GGML_TYPE_ROW4_CODES_PAIR2) {
        return ggml_are_same_shape(op->src[0], op) && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op);
    }

    // custom complex types are CPU-only for compute ops, except when they are stored as leaf tensors for fused kernels.
    if (op->op != GGML_OP_NONE &&
        (op->type == GGML_TYPE_IFAIRY || op->type == GGML_TYPE_IFAIRY_Q16 || op->type == GGML_TYPE_IFAIRY64 ||
         op->type == GGML_TYPE_FAIRY2I_TILE64_V2 || op->type == GGML_TYPE_FAIRY2I_ACT_Q16_64 ||
         op->type == GGML_TYPE_FAIRY2I_BUNDLE_CODES || op->type == GGML_TYPE_ROW4_CODES ||
         op->type == GGML_TYPE_ROW4_CODES_PAIR2)) {
        return false;
    }
    for (size_t i = 0, n = GGML_MAX_SRC; i < n; ++i) {
        if (op->src[i] != NULL &&
            (op->src[i]->type == GGML_TYPE_IFAIRY || op->src[i]->type == GGML_TYPE_IFAIRY_Q16 ||
             op->src[i]->type == GGML_TYPE_IFAIRY64 || op->src[i]->type == GGML_TYPE_FAIRY2I_TILE64_V2 ||
             op->src[i]->type == GGML_TYPE_FAIRY2I_ACT_Q16_64 ||
             op->src[i]->type == GGML_TYPE_FAIRY2I_BUNDLE_CODES ||
             op->src[i]->type == GGML_TYPE_ROW4_CODES || op->src[i]->type == GGML_TYPE_ROW4_CODES_PAIR2)) {
            return false;
        }
    }

    if (!has_bfloat) {
        if (op->type == GGML_TYPE_BF16) {
            return false;
        }

        for (size_t i = 0, n = 3; i < n; ++i) {
            if (op->src[i] != NULL && op->src[i]->type == GGML_TYPE_BF16) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_EXP:
                    return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    return ggml_is_contiguous_1(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
               default:
                    return false;
            }
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
        case GGML_OP_CONCAT:
            return true;
        case GGML_OP_COMPLEX_SPLIT:
        case GGML_OP_COMPLEX_MERGE:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op);
        case GGML_OP_COMPLEX_ADD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]) &&
                   ggml_is_contiguous_rows(op->src[1]) && ggml_is_contiguous_rows(op);
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_ADD_ID:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_ACC:
        case GGML_OP_REPEAT:
        case GGML_OP_SCALE:
        case GGML_OP_CONV_TRANSPOSE_1D:
            return true;
        case GGML_OP_CLAMP:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_GROUP_NORM:
            return has_simdgroup_reduction && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return has_simdgroup_reduction && (op->ne[0] % 4 == 0 && ggml_is_contiguous_1(op->src[0]));
        case GGML_OP_FAIRY2I_RMS_NORM_EXACT:
            return has_bfloat && op->src[0] && op->src[1] && op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                   op->src[0]->nb[0] == sizeof(float) && op->src[1]->nb[0] == sizeof(float) &&
                   op->nb[0] == sizeof(float) && ggml_are_same_shape(op->src[0], op) &&
                   ggml_can_repeat(op->src[1], op->src[0]);
        case GGML_OP_FAIRY2I_SILU_EXACT:
            return has_bfloat && op->src[0] && op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 && ggml_are_same_shape(op->src[0], op) &&
                   op->src[0]->nb[0] == sizeof(float) && ggml_is_contiguous_1(op->src[0]) &&
                   ggml_is_contiguous(op);
        case GGML_OP_FAIRY2I_MUL_EXACT:
            return has_bfloat && op->src[0] && op->src[1] && op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_are_same_shape(op->src[0], op->src[1]) && ggml_are_same_shape(op->src[0], op) &&
                   op->src[0]->nb[0] == sizeof(float) && op->src[1]->nb[0] == sizeof(float) &&
                   ggml_is_contiguous_1(op->src[0]) && ggml_is_contiguous_1(op->src[1]) && ggml_is_contiguous(op);
        case GGML_OP_FAIRY2I_PACK_BF16_EXACT:
            return has_bfloat && op->src[0] && (op->type == GGML_TYPE_BF16 || op->type == GGML_TYPE_F32) &&
                   op->src[0]->type == GGML_TYPE_F32 && ggml_are_same_shape(op->src[0], op) &&
                   ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op);
        case GGML_OP_FAIRY2I_ATTN_EXACT_CPU:
            return false;
        case GGML_OP_ARGMAX:
            return has_simdgroup_reduction;
        case GGML_OP_NORM:
            return has_simdgroup_reduction && (op->ne[0] % 4 == 0 && ggml_is_contiguous_1(op->src[0]));
        case GGML_OP_ROPE:
            return true;
        case GGML_OP_FAIRY2I_ROPE_EXACT:
            {
                const int mode   = ggml_get_op_params_i32(op, 2);
                const int n_dims = ggml_get_op_params_i32(op, 1);
                return has_bfloat && op->src[0] && op->src[1] && op->type == GGML_TYPE_F32 &&
                       op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_I32 &&
                       (!op->src[2] || op->src[2]->type == GGML_TYPE_F32) &&
                       op->src[0]->nb[0] == sizeof(float) && op->src[1]->nb[0] == sizeof(int32_t) &&
                       (!op->src[2] ||
                        (op->src[2]->nb[0] == sizeof(float) && op->src[2]->ne[0] >= n_dims / 2)) &&
                       op->nb[0] == sizeof(float) && ggml_are_same_shape(op->src[0], op) &&
                       op->src[1]->ne[0] >= op->src[0]->ne[2] &&
                       op->src[1]->ne[0] % op->src[0]->ne[2] == 0 &&
                       (mode & GGML_ROPE_TYPE_NEOX) != 0 && (mode & GGML_ROPE_TYPE_MROPE) == 0 &&
                       mode != GGML_ROPE_TYPE_VISION && n_dims > 0 && n_dims <= op->src[0]->ne[0] &&
                       n_dims % 2 == 0 &&
                       GGML_PAD((size_t) n_dims * sizeof(float), 16) <= dev->props.max_theadgroup_memory_size;
            }
        case GGML_OP_IM2COL:
            return ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32 && (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32);
        case GGML_OP_POOL_1D:
            return false;
        case GGML_OP_UPSCALE:
            return op->src[0]->type == GGML_TYPE_F32 && op->op_params[0] == GGML_SCALE_MODE_NEAREST;
        case GGML_OP_POOL_2D:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_PAD:
            return (ggml_get_op_params_i32(op, 0) == 0) && (ggml_get_op_params_i32(op, 2) == 0) &&
                   (ggml_get_op_params_i32(op, 4) == 0) && (ggml_get_op_params_i32(op, 6) == 0);
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_ARGSORT:
        case GGML_OP_LEAKY_RELU:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_ARANGE:
            return true;
        case GGML_OP_FLASH_ATTN_EXT:
            if (ggml_flash_attn_ext_get_fairy2i_exact(op) && !ggml_flash_attn_ext_get_fairy2i_flash3(op)) {
                const struct ggml_tensor * q    = op->src[0];
                const struct ggml_tensor * k    = op->src[1];
                const struct ggml_tensor * v    = op->src[2];
                const struct ggml_tensor * mask = op->src[3];

                if (!q || !k || !v || !mask || op->src[4]) {
                    return false;
                }

                const float max_bias      = ggml_get_op_params_f32(op, 1);
                const float logit_softcap = ggml_get_op_params_f32(op, 2);
                const bool supported_dims =
                    (q->ne[0] == v->ne[0] &&
                     (q->ne[0] == 40 || q->ne[0] == 64 || q->ne[0] == 80 ||
                      q->ne[0] == 96 || q->ne[0] == 112 || q->ne[0] == 128 ||
                      q->ne[0] == 192 || q->ne[0] == 256)) ||
                    (q->ne[0] == 192 && v->ne[0] == 128);

                return has_bfloat && has_simdgroup_mm && has_simdgroup_reduction &&
                       op->type == GGML_TYPE_F32 && q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_BF16 &&
                       v->type == GGML_TYPE_BF16 && mask->type == GGML_TYPE_F32 &&
                       max_bias == 0.0f && logit_softcap == 0.0f && supported_dims &&
                       q->ne[0] == k->ne[0] && k->ne[1] == v->ne[1] &&
                       k->ne[2] == v->ne[2] && k->ne[3] == v->ne[3] &&
                       q->ne[2] % k->ne[2] == 0 && q->ne[3] == k->ne[3] &&
                       k->ne[1] % 32 == 0 && mask->ne[0] >= k->ne[1] &&
                       mask->ne[1] >= GGML_PAD(q->ne[1], GGML_KQ_MASK_PAD) &&
                       q->ne[2] % mask->ne[2] == 0 && q->ne[3] % mask->ne[3] == 0 &&
                       q->nb[0] == sizeof(float) && k->nb[0] == sizeof(ggml_bf16_t) &&
                       v->nb[0] == sizeof(ggml_bf16_t) && mask->nb[0] == sizeof(float) &&
                       op->nb[0] == sizeof(float) && ggml_is_contiguous_rows(q) &&
                       ggml_is_contiguous_rows(k) && ggml_is_contiguous_rows(v) &&
                       ggml_is_contiguous(mask) && ggml_is_contiguous(op);
            }

            // for new head sizes, add checks here
            if (op->src[0]->ne[0] != 40 &&
                op->src[0]->ne[0] != 64 &&
                op->src[0]->ne[0] != 80 &&
                op->src[0]->ne[0] != 96 &&
                op->src[0]->ne[0] != 112 &&
                op->src[0]->ne[0] != 128 &&
                op->src[0]->ne[0] != 192 &&
                op->src[0]->ne[0] != 256) {
                return false;
            }
            if (op->src[0]->ne[0] == 576) {
                // DeepSeek sizes
                // TODO: disabled for now, until optmized
                return false;
            }
            if (op->src[1]->type != op->src[2]->type) {
                return false;
            }
            return has_simdgroup_mm; // TODO: over-restricted for vec-kernels
        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN:
            return has_simdgroup_reduction;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            return has_simdgroup_reduction &&
                (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F32);
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        switch (op->type) {
                           case GGML_TYPE_F32:
                           case GGML_TYPE_F16:
                           case GGML_TYPE_BF16:
                           case GGML_TYPE_Q8_0:
                           case GGML_TYPE_Q4_0:
                           case GGML_TYPE_Q4_1:
                           case GGML_TYPE_Q5_0:
                           case GGML_TYPE_Q5_1:
                           case GGML_TYPE_IQ4_NL:
                           case GGML_TYPE_I32:
                                return true;
                           default:
                                return false;
                        }
                    case GGML_TYPE_F16:
                        switch (op->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_F16:
                                return true;
                            default:
                                return false;
                        }
                    case GGML_TYPE_BF16:
                        switch (op->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_BF16:
                                return true;
                            default:
                                return false;
                        }
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                        switch (op->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_F16:
                                return true;
                            default:
                                return false;
                        }
                    case GGML_TYPE_I32:
                        return op->type == GGML_TYPE_F32;
                    default:
                        return false;
                };
            }
        case GGML_OP_GET_ROWS:
            {
                return op->ne[3] == 1;
            }
        case GGML_OP_SET_ROWS:
            {
                const int32_t mode = ggml_get_op_params_i32(op, 0);
                if (mode == GGML_SET_ROWS_BF16_CARRIER_ROWS) {
                    return op->type == GGML_TYPE_BF16 && op->src[0]->type == GGML_TYPE_F32 &&
                           op->src[1]->type == GGML_TYPE_I64 && op->ne[0] == op->src[0]->ne[0] &&
                           op->ne[2] == 1 && op->ne[3] == 1 && op->src[0]->ne[1] == op->src[1]->ne[0] &&
                           op->src[0]->ne[2] == 1 && op->src[0]->ne[3] == 1 && op->src[1]->ne[1] == 1 &&
                           op->src[1]->ne[2] == 1 && op->src[1]->ne[3] == 1 &&
                           op->nb[0] == sizeof(ggml_bf16_t) && op->src[0]->nb[0] == sizeof(float) &&
                           op->src[1]->nb[0] == sizeof(int64_t) && ggml_is_contiguous_rows(op);
                }
                if (mode == GGML_SET_ROWS_BF16_CARRIER_ELEMENTS) {
                    return op->type == GGML_TYPE_BF16 && op->src[0]->type == GGML_TYPE_F32 &&
                           op->src[1]->type == GGML_TYPE_I64 && op->ne[0] == 1 && op->ne[2] == 1 && op->ne[3] == 1 &&
                           op->src[0]->ne[3] == 1 && op->src[1]->ne[0] == ggml_nelements(op->src[0]) &&
                           op->src[1]->ne[1] == 1 && op->src[1]->ne[2] == 1 && op->src[1]->ne[3] == 1 &&
                           op->nb[0] == sizeof(ggml_bf16_t) && op->src[0]->nb[0] == sizeof(float) &&
                           op->src[1]->nb[0] == sizeof(int64_t) && ggml_is_contiguous_rows(op);
                }
                if (mode != 0) {
                    return false;
                }
                if (op->src[0]->type == GGML_TYPE_BF16) {
                    return op->type == GGML_TYPE_BF16;
                }
                if (op->src[0]->type != GGML_TYPE_F32) {
                    return false;
                }

                switch (op->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_IQ4_NL:
                        return true;
                    default:
                        return false;
                };
            }
        default:
            return false;
    }
}

const struct ggml_metal_device_props * ggml_metal_device_get_props(ggml_metal_device_t dev) {
    return &dev->props;
}

//
// device buffers
//

// max memory buffers that can be mapped to the device
#define GGML_METAL_MAX_BUFFERS 64

struct ggml_metal_buffer_wrapper {
    void   * data;
    size_t   size;

    id<MTLBuffer> metal;
};

struct ggml_metal_fairy2i_w1_coeff_lut {
    const void * tensor_data;
    size_t tensor_size;
    id<MTLBuffer> metal;

    struct ggml_metal_fairy2i_w1_coeff_lut * next;
};

struct ggml_metal_buffer {
    void * all_data; // TODO: https://github.com/ggml-org/llama.cpp/pull/15985
    size_t all_size;

    // if false, the Metal buffer data is allocated in private GPU memory and is not shared with the host
    bool is_shared;
    bool owned;

    // multiple buffers are used only to avoid the maximum buffer size limitation when using mmap
    int n_buffers;
    struct ggml_metal_buffer_wrapper buffers[GGML_METAL_MAX_BUFFERS];

    bool use_residency_sets;

    // optional MTLResidencySet
    // note: cannot use explicity "id<MTLResidencySet>" here because it is not available on certain OSes
    id rset;

    // pointers to global device objects
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;

    struct ggml_metal_fairy2i_w1_coeff_lut * fairy2i_w1_coeff_luts;
};

static void ggml_metal_log_allocated_size(id<MTLDevice> device, size_t size_aligned) {
#ifndef GGML_METAL_NDEBUG
#if TARGET_OS_OSX || (TARGET_OS_IOS && __clang_major__ >= 15)
    if (@available(macOS 10.12, iOS 16.0, *)) {
        GGML_LOG_DEBUG("%s: allocated buffer, size = %8.2f MiB, (%8.2f / %8.2f)\n",
                __func__,
                size_aligned / 1024.0 / 1024.0,
                device.currentAllocatedSize / 1024.0 / 1024.0,
                device.recommendedMaxWorkingSetSize / 1024.0 / 1024.0);

        if (device.currentAllocatedSize > device.recommendedMaxWorkingSetSize) {
            GGML_LOG_WARN("%s: warning: current allocated size is greater than the recommended max working set size\n", __func__);
        }
    } else {
        GGML_LOG_INFO("%s: allocated buffer, size = %8.2f MiB, (%8.2f)\n",
                __func__,
                size_aligned / 1024.0 / 1024.0,
                device.currentAllocatedSize / 1024.0 / 1024.0);
    }
#endif
#endif
    GGML_UNUSED(device);
    GGML_UNUSED(size_aligned);
}

// rset init
static bool ggml_metal_buffer_rset_init(ggml_metal_buffer_t buf) {
    buf->rset = nil;

    if (!buf->use_residency_sets) {
        return true;
    }

#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        MTLResidencySetDescriptor * desc = [[MTLResidencySetDescriptor alloc] init];
        desc.label = @"ggml_metal";
        desc.initialCapacity = buf->n_buffers;

        NSError * error;
        buf->rset = [buf->device newResidencySetWithDescriptor:desc error:&error];
        if (error) {
            GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
            [desc release];
            return false;
        }

        [desc release];

        for (int i = 0; i < buf->n_buffers; i++) {
            [buf->rset addAllocation:buf->buffers[i].metal];
        }

        [buf->rset commit];
        [buf->rset requestResidency];

        return true;
    }
#endif

    return true;
}

// rset free
static void ggml_metal_buffer_rset_free(ggml_metal_buffer_t buf) {
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        if (buf->rset) {
            [buf->rset endResidency];
            [buf->rset removeAllAllocations];
            [buf->rset release];
        }
    }
#else
    GGML_UNUSED(buf);
#endif
}

static void * ggml_metal_host_malloc(size_t n) {
    void * data = NULL;

#if TARGET_OS_OSX
    kern_return_t err = vm_allocate((vm_map_t) mach_task_self(), (void *) &data, n, VM_FLAGS_ANYWHERE);
    if (err != KERN_SUCCESS) {
        GGML_LOG_ERROR("%s: error: vm_allocate failed\n", __func__);
        return NULL;
    }
#else
    const int result = posix_memalign((void **) &data, sysconf(_SC_PAGESIZE), n);
    if (result != 0) {
        GGML_LOG_ERROR("%s: error: posix_memalign failed\n", __func__);
        return NULL;
    }
#endif

    return data;
}

ggml_metal_buffer_t ggml_metal_buffer_init(ggml_metal_device_t dev, size_t size, bool shared) {
    ggml_metal_buffer_t res = calloc(1, sizeof(struct ggml_metal_buffer));

    const size_t size_page = sysconf(_SC_PAGESIZE);

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

    shared = shared && props_dev->use_shared_buffers;

    // allocate shared buffer if the device supports it and it is required by the buffer type
    if (shared) {
        res->all_data = ggml_metal_host_malloc(size_aligned);
        res->is_shared = true;
        res->owned = true;
    } else {
        // dummy, non-NULL value - we'll populate this after creating the Metal buffer below
        res->all_data = (void *) 0x000000400ULL;
        res->is_shared = false;
    }
    res->all_size = size_aligned;

    res->device = ggml_metal_device_get_obj(dev);
    res->queue  = ggml_metal_device_get_queue(dev);

    res->n_buffers = 1;

    if (res->all_data != NULL) {
        res->buffers[0].size  = size;
        res->buffers[0].metal = nil;

        if (size_aligned > 0) {
            if (props_dev->use_shared_buffers &&shared) {
                res->buffers[0].metal = [res->device newBufferWithBytesNoCopy:res->all_data
                                                                  length:size_aligned
                                                                 options:MTLResourceStorageModeShared
                                                             deallocator:nil];
            } else {
                res->buffers[0].metal = [res->device newBufferWithLength:size_aligned options:MTLResourceStorageModePrivate];

                res->all_data = (void *) (res->buffers[0].metal.gpuAddress);
            }
        }

        res->buffers[0].data = res->all_data;
    }

    if (size_aligned > 0 && (res->all_data == NULL || res->buffers[0].metal == nil)) {
        GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_aligned / 1024.0 / 1024.0);
        free(res);
        return NULL;
    }

    res->use_residency_sets = props_dev->use_residency_sets;

    if (!ggml_metal_buffer_rset_init(res)) {
        GGML_LOG_ERROR("%s: error: failed to initialize residency set\n", __func__);
        free(res);
        return NULL;
    }

    //ggml_metal_log_allocated_size(device, size_aligned);

    return res;
}

ggml_metal_buffer_t ggml_metal_buffer_map(ggml_metal_device_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    ggml_metal_buffer_t res = calloc(1, sizeof(struct ggml_metal_buffer));

    res->all_data = ptr;
    res->all_size = size;

    res->is_shared = true;
    res->owned = false;

    res->n_buffers = 0;

    const size_t size_page = sysconf(_SC_PAGESIZE);

    // page-align the data ptr
    {
        const uintptr_t offs = (uintptr_t) ptr % size_page;
        ptr  = (void *) ((char *) ptr - offs);
        size += offs;
    }

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    res->device = ggml_metal_device_get_obj(dev);
    res->queue  = ggml_metal_device_get_queue(dev);

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

    // the buffer fits into the max buffer size allowed by the device
    if (size_aligned <= props_dev->max_buffer_size) {
        res->buffers[res->n_buffers].data  = ptr;
        res->buffers[res->n_buffers].size  = size;
        res->buffers[res->n_buffers].metal = nil;

        if (size_aligned > 0) {
            res->buffers[res->n_buffers].metal = [res->device newBufferWithBytesNoCopy:ptr length:size_aligned options:MTLResourceStorageModeShared deallocator:nil];

            if (res->buffers[res->n_buffers].metal == nil) {
                GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_aligned / 1024.0 / 1024.0);
                free(res);
                return NULL;
            }
        }

        ggml_metal_log_allocated_size(res->device, size_aligned);

        ++res->n_buffers;
    } else {
        // this overlap between the views will guarantee that the tensor with the maximum size will fully fit into
        // one of the views
        const size_t size_ovlp = ((max_tensor_size + size_page - 1) / size_page + 1) * size_page; // round-up 2 pages just in case
        const size_t size_step = props_dev->max_buffer_size - size_ovlp;
        const size_t size_view = props_dev->max_buffer_size;

        for (size_t i = 0; i < size; i += size_step) {
            const size_t size_step_aligned = (i + size_view <= size) ? size_view : (size_aligned - i);

            res->buffers[res->n_buffers].data  = (void *) ((uint8_t *) ptr + i);
            res->buffers[res->n_buffers].size  = size_step_aligned;
            res->buffers[res->n_buffers].metal = nil;

            if (size_step_aligned > 0) {
                res->buffers[res->n_buffers].metal = [res->device newBufferWithBytesNoCopy:(void *) ((uint8_t *) ptr + i) length:size_step_aligned options:MTLResourceStorageModeShared deallocator:nil];

                if (res->buffers[res->n_buffers].metal == nil) {
                    GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_step_aligned / 1024.0 / 1024.0);
                    free(res);
                    return NULL;
                }
            }

            ggml_metal_log_allocated_size(res->device, size_step_aligned);

            if (i + size_step < size) {
                GGML_LOG_INFO("\n");
            }

            ++res->n_buffers;
        }
    }

    res->use_residency_sets = props_dev->use_residency_sets;

    if (!ggml_metal_buffer_rset_init(res)) {
        GGML_LOG_ERROR("%s: error: failed to initialize residency set\n", __func__);
        free(res);
        return NULL;
    }

    return res;
}

void ggml_metal_buffer_free(ggml_metal_buffer_t buf) {
    struct ggml_metal_fairy2i_w1_coeff_lut * lut = buf->fairy2i_w1_coeff_luts;
    while (lut) {
        struct ggml_metal_fairy2i_w1_coeff_lut * next = lut->next;
        [lut->metal release];
        free(lut);
        lut = next;
    }

    for (int i = 0; i < buf->n_buffers; i++) {
        [buf->buffers[i].metal release];
    }

    ggml_metal_buffer_rset_free(buf);

    if (buf->is_shared && buf->owned) {
#if TARGET_OS_OSX
        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)buf->all_data, buf->all_size);
#else
        free(buf->all_data);
#endif
    }

    free(buf);
}

void * ggml_metal_buffer_get_base(ggml_metal_buffer_t buf) {
    return buf->all_data;
}

bool ggml_metal_buffer_is_shared(ggml_metal_buffer_t buf) {
    return buf->is_shared;
}

static bool ggml_metal_fairy2i_is_exact_w1_bundle_scales(const struct ggml_tensor * tensor) {
    return tensor->type == GGML_TYPE_BF16 && tensor->ne[0] == 2 && tensor->ne[1] == 2 && tensor->ne[2] > 0 &&
           tensor->ne[3] == 1 && ggml_is_contiguous(tensor);
}

static uint16_t ggml_metal_fairy2i_add_bf16_rne(uint16_t a, uint16_t b) {
    const ggml_bf16_t a_bf16 = { a };
    const ggml_bf16_t b_bf16 = { b };
    return ggml_fp32_to_bf16(ggml_bf16_to_fp32(a_bf16) + ggml_bf16_to_fp32(b_bf16)).bits;
}

static uint8_t ggml_metal_fairy2i_bf16_product_metric(uint16_t value) {
    const uint16_t abs_value = value & 0x7fff;
    if (abs_value == 0) {
        return 255;
    }
    const uint16_t exponent = abs_value >> 7;
    return exponent == 0 || exponent == 0xff ? 0 : exponent;
}

static uint8_t ggml_metal_fairy2i_one_add_coefficient_metric_bound(uint8_t min_metric, uint8_t max_exponent) {
    if (min_metric == 255) {
        return 255;
    }
    if (min_metric <= 7 || max_exponent >= 254) {
        return 0;
    }
    return min_metric - 7;
}

static void ggml_metal_fairy2i_build_exact_w1_coeff_lut(
        uint16_t * dst,
        const uint16_t * scales,
        size_t tiles) {
    const size_t entries = tiles * 16;
    uint8_t * metrics = (uint8_t *) (dst + entries * 4);
    uint8_t * tile_metrics = metrics + entries;
    for (size_t tile = 0; tile < tiles; ++tile) {
        const uint16_t * scale = scales + tile * 4;
        uint8_t min_scale_metric = 255;
        uint8_t max_scale_exponent = 0;
        for (int component = 0; component < 4; ++component) {
            const uint8_t metric = ggml_metal_fairy2i_bf16_product_metric(scale[component]);
            min_scale_metric = MIN(min_scale_metric, metric);
            max_scale_exponent = MAX(max_scale_exponent, metric == 255 ? 0 : metric);
        }
        tile_metrics[tile] =
            ggml_metal_fairy2i_one_add_coefficient_metric_bound(min_scale_metric, max_scale_exponent);
        for (uint32_t pattern = 0; pattern < 16; ++pattern) {
            const uint32_t u_code = pattern & 3u;
            const uint32_t w_code = pattern >> 2;
            const uint16_t u_scale = scale[(u_code & 2u) == 0u ? 0 : 1];
            const uint16_t w_scale = scale[(w_code & 2u) == 0u ? 2 : 3];
            const uint16_t u_stage = u_scale ^ ((u_code & 1u) != 0u ? 0x0000u : 0x8000u);
            const uint16_t w_stage = w_scale ^ ((w_code & 1u) != 0u ? 0x0000u : 0x8000u);
            const uint16_t u_real = (u_code & 2u) == 0u ? u_stage : 0;
            const uint16_t u_imag = (u_code & 2u) == 0u ? 0 : u_stage;
            const uint16_t w_real = (w_code & 2u) == 0u ? w_stage : 0;
            const uint16_t w_imag = (w_code & 2u) == 0u ? 0 : w_stage;
            uint16_t * coeff = dst + (tile * 16 + pattern) * 4;
            coeff[0] = ggml_metal_fairy2i_add_bf16_rne(u_real, w_real);
            coeff[1] = ggml_metal_fairy2i_add_bf16_rne(u_imag ^ 0x8000u, w_imag);
            coeff[2] = ggml_metal_fairy2i_add_bf16_rne(u_imag, w_imag);
            coeff[3] = ggml_metal_fairy2i_add_bf16_rne(u_real, w_real ^ 0x8000u);
            metrics[tile * 16 + pattern] = MIN(
                MIN(ggml_metal_fairy2i_bf16_product_metric(coeff[0]),
                    ggml_metal_fairy2i_bf16_product_metric(coeff[1])),
                MIN(ggml_metal_fairy2i_bf16_product_metric(coeff[2]),
                    ggml_metal_fairy2i_bf16_product_metric(coeff[3])));
        }
    }
}

static void ggml_metal_fairy2i_invalidate_w1_coeff_lut(
        ggml_metal_buffer_t buf,
        const struct ggml_tensor * tensor) {
    struct ggml_metal_fairy2i_w1_coeff_lut ** next = &buf->fairy2i_w1_coeff_luts;
    while (*next) {
        struct ggml_metal_fairy2i_w1_coeff_lut * lut = *next;
        if (lut->tensor_data == tensor->data) {
            *next = lut->next;
            [lut->metal release];
            free(lut);
            continue;
        }
        next = &lut->next;
    }
}

size_t ggml_metal_fairy2i_packed_weight_extra(const struct ggml_tensor * tensor) {
    GGML_UNUSED(tensor);
    return 0;
}

void ggml_metal_buffer_memset_tensor(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    @synchronized (buf->buffers[0].metal) {
        ggml_metal_fairy2i_invalidate_w1_coeff_lut(buf, tensor);
    }

    if (buf->is_shared) {
        memset((char *)tensor->data + offset, value, size);
        return;
    }

    @autoreleasepool {
        // dst
        struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(buf, tensor);
        bid_dst.offs += offset;

        id<MTLCommandQueue>  queue   = buf->queue;
        id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder fillBuffer:bid_dst.metal
                          range:NSMakeRange(bid_dst.offs, bid_dst.offs + size)
                          value:value];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

void ggml_metal_buffer_set_tensor(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    @synchronized (buf->buffers[0].metal) {
        ggml_metal_fairy2i_invalidate_w1_coeff_lut(buf, tensor);
    }

    if (buf->is_shared) {
        memcpy((char *)tensor->data + offset, data, size);
        return;
    }

    @autoreleasepool {
        // src
        void * data_ptr = (void *)(uintptr_t) data; // "const cast" the src data
        id<MTLBuffer> buf_src = [buf->device newBufferWithBytesNoCopy:data_ptr
                                                               length:size
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];

        // dst
        struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(buf, tensor);
        bid_dst.offs += offset;

        // note: for experimentation purposes, here we use a semaphore to wait for the copy to complete
        //       this is alternative to waitUntilCompleted, which should be faster, but don't seem to make much difference
        dispatch_semaphore_t completion_semaphore = dispatch_semaphore_create(0);

        id<MTLCommandQueue>  queue   = buf->queue;
        id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder copyFromBuffer:buf_src
                       sourceOffset:0
                           toBuffer:bid_dst.metal
                  destinationOffset:bid_dst.offs
                               size:size];

            [encoder endEncoding];
        }

        [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                             // TODO: can check for errors here
            GGML_UNUSED(cb);

            dispatch_semaphore_signal(completion_semaphore);
        }];

        [cmd_buf commit];

        dispatch_semaphore_wait(completion_semaphore, DISPATCH_TIME_FOREVER);
        dispatch_release(completion_semaphore);

        //[cmd_buf waitUntilCompleted];
    }
}

void ggml_metal_buffer_get_tensor(ggml_metal_buffer_t buf, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (buf->is_shared) {
        memcpy(data, (const char *)tensor->data + offset, size);
        return;
    }

    @autoreleasepool {
        // src
        struct ggml_metal_buffer_id bid_src = ggml_metal_buffer_get_id(buf, tensor);
        bid_src.offs += offset;

        // dst
        id<MTLBuffer> buf_dst = [buf->device newBufferWithBytesNoCopy:data
                                                               length:size
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];

        id<MTLCommandQueue>  queue   = buf->queue;
        id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder copyFromBuffer:bid_src.metal
                       sourceOffset:bid_src.offs
                           toBuffer:buf_dst
                  destinationOffset:0
                               size:size];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

void ggml_metal_buffer_clear(ggml_metal_buffer_t buf, uint8_t value) {
    @synchronized (buf->buffers[0].metal) {
        struct ggml_metal_fairy2i_w1_coeff_lut * lut = buf->fairy2i_w1_coeff_luts;
        while (lut) {
            struct ggml_metal_fairy2i_w1_coeff_lut * next = lut->next;
            [lut->metal release];
            free(lut);
            lut = next;
        }
        buf->fairy2i_w1_coeff_luts = NULL;
    }

    if (buf->is_shared) {
        memset(buf->all_data, value, buf->all_size);
        return;
    }

    @autoreleasepool {
        id<MTLCommandQueue>  queue   = buf->queue;
        id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder fillBuffer:buf->buffers[0].metal
                          range:NSMakeRange(0, buf->buffers[0].size)
                          value:value];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

struct ggml_metal_buffer_id ggml_metal_buffer_get_id(ggml_metal_buffer_t buf, const struct ggml_tensor * t) {
    struct ggml_metal_buffer_id res = { nil, 0 };

    const int64_t tsize = ggml_nbytes(t);

    // find the view that contains the tensor fully
    for (int i = 0; i < buf->n_buffers; ++i) {
        const int64_t ioffs = (int64_t) t->data - (int64_t) buf->buffers[i].data;

        //GGML_LOG_INFO("ioffs = %10ld, tsize = %10ld, sum = %10ld, buf->buffers[%d].size = %10ld\n", ioffs, tsize, ioffs + tsize, i, buf->buffers[i].size);
        if (ioffs >= 0 && ioffs + tsize <= (int64_t) buf->buffers[i].size) {
            res.metal = buf->buffers[i].metal;
            res.offs  = (size_t) ioffs;

            //GGML_LOG_INFO("%s: tensor '%16s', offs = %8ld\n", __func__, t->name, *offs);

            return res;
        }
    }

    GGML_LOG_ERROR("%s: error: tensor '%s' buffer is nil\n", __func__, t->name);

    return res;
}

struct ggml_metal_buffer_id ggml_metal_buffer_get_fairy2i_w1_coeff_lut(
        ggml_metal_buffer_t buf,
        const struct ggml_tensor * scales) {
    GGML_ASSERT(ggml_metal_fairy2i_is_exact_w1_bundle_scales(scales));

    @synchronized (buf->buffers[0].metal) {
        for (struct ggml_metal_fairy2i_w1_coeff_lut * lut = buf->fairy2i_w1_coeff_luts; lut; lut = lut->next) {
            if (lut->tensor_data == scales->data && lut->tensor_size == ggml_nbytes(scales)) {
                return (struct ggml_metal_buffer_id) { lut->metal, 0 };
            }
        }

        const size_t tiles = scales->ne[2];
        const size_t scale_size = ggml_nbytes(scales);
        const size_t lut_size = tiles * 16 * (4 * sizeof(uint16_t) + sizeof(uint8_t)) + tiles * sizeof(uint8_t);
        uint16_t * scale_data = malloc(scale_size);
        uint16_t * lut_data = malloc(lut_size);
        GGML_ASSERT(scale_data != NULL);
        GGML_ASSERT(lut_data != NULL);

        ggml_metal_buffer_get_tensor(buf, scales, scale_data, 0, scale_size);
        ggml_metal_fairy2i_build_exact_w1_coeff_lut(lut_data, scale_data, tiles);

        id<MTLBuffer> metal = [buf->device newBufferWithBytes:lut_data
                                                      length:lut_size
                                                     options:MTLResourceStorageModeShared];
        GGML_ASSERT(metal != nil);

        free(lut_data);
        free(scale_data);

        struct ggml_metal_fairy2i_w1_coeff_lut * lut = calloc(1, sizeof(*lut));
        GGML_ASSERT(lut != NULL);
        lut->tensor_data = scales->data;
        lut->tensor_size = scale_size;
        lut->metal = metal;
        lut->next = buf->fairy2i_w1_coeff_luts;
        buf->fairy2i_w1_coeff_luts = lut;

        ggml_metal_log_allocated_size(buf->device, lut_size);
        return (struct ggml_metal_buffer_id) { metal, 0 };
    }
}

struct ggml_metal_buffer_id ggml_metal_device_get_scratch(ggml_metal_device_t dev, size_t size) {
    const size_t size_page = sysconf(_SC_PAGESIZE);
    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += size_page - (size_aligned % size_page);
    }

    if (dev->scratch_size < size_aligned) {
        ggml_metal_buffer_t scratch = ggml_metal_buffer_init(dev, size_aligned, false);
        if (scratch == NULL) {
            GGML_LOG_ERROR("%s: error: failed to allocate scratch buffer, size = %8.2f MiB\n", __func__,
                           size_aligned / 1024.0 / 1024.0);
            return (struct ggml_metal_buffer_id) { nil, 0 };
        }

        if (dev->scratch) {
            ggml_metal_buffer_free(dev->scratch);
        }
        dev->scratch      = scratch;
        dev->scratch_size = size_aligned;
    }

    return (struct ggml_metal_buffer_id) { dev->scratch->buffers[0].metal, 0 };
}
