#include <core/Renderer.h>

#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include <hdr/HDRImage.h>
#include <hdr/HDRKernel.cuh>
#include <math/CutilMath.h>
#include <geometry/IntersectKernel.cuh>
#include <geometry/Ray.h>
#include <geometry/GeometryCompact.h>
#include <geometry/SphereImpl.cuh>
#include <geometry/PlaneImpl.cuh>
#include <material/CoatImpl.cuh>
#include <material/DiffImpl.cuh>
#include <material/MetalImpl.cuh>
#include <material/SpecImpl.cuh>
#include <material/RefrImpl.cuh>

// union struct required for mapping pixel colours to OpenGL buffer
union Color  // 4 bytes = 4 chars = 1 float
{
    float c;
    uchar4 components;
};

__device__ __inline__ Vec3f renderKernel(curandState *randState, HDRImage *hdrEnv, BVHCompact *bvhCompact,
                                         GeometryCompact *geometryCompact, MaterialCompact *materialCompact, Ray ray,
                                         RenderMeta *renderMeta) {

    Vec3f mask = Vec3f(1.0f, 1.0f, 1.0f); // colour mask
    Vec3f accumulatedColor = Vec3f(0.0f, 0.0f, 0.0f); // accumulated colour
    Vec3f emit;

    for (int bounces = 0; bounces < 10; bounces++) {
        // iteration up to 4 bounces (instead of recursion in CPU code)
        Hit hit = intersect(ray, bvhCompact, true);
        Hit geometryHit;
        geometryHit.distance = ray.tMax;
        for (int i = 0; i < geometryCompact->geometriesSize; i++) {
            Geometry *geometry = (Geometry *) (geometryCompact->geometries + i);
            switch (geometry->type) {
                case SPHERE:
                    geometryHit = sphereIntersect((Sphere *) geometry, ray);
                    break;
                case CUBE:
                    break;
                case PLANE:
                    geometryHit = planeIntersect((Plane *) geometry, ray);
                    break;
                case MESH:
                    break;
            }

            if (geometryHit.distance < ray.tMax) {
                geometryHit.index = i;
                if (geometryHit < hit) {
                    geometryHit.matIndex = tex1Dfetch<int>(geometryCompact->matIndicesTexture, geometryHit.index);
                    hit = geometryHit;
                }
            }
        }


        if (hit.distance > 1e19) {
            emit = hdrSample(hdrEnv, ray, renderMeta);
            accumulatedColor += (mask * emit);
            return accumulatedColor;
        }

        normalize(hit, ray);
        hitPoint(hit, ray);

        Material *material = (Material *) (materialCompact->materials + hit.matIndex);
        accumulatedColor += mask * material->emission;
        switch (material->type) {
            case COAT:
                ray = coatSample((Coat *) material, randState, ray, hit, mask);
                break;
            case DIFF:
                ray = diffSample((Diff *) material, randState, ray, hit, mask);
                break;
            case METAL:
                ray = metalSample((Metal *) material, randState, ray, hit, mask);
                break;
            case SPEC:
                ray = specSample((Spec *) material, randState, ray, hit, mask);
                break;
            case REFR:
                ray = refrSample((Refr *) material, randState, ray, hit, mask);
                break;
        }
    }
    return accumulatedColor;
}

__global__ void pathTracingKernel(Vec3f *outputBuffer, Vec3f *accumulatedBuffer, HDRImage *hdrEnv,
                                  RenderMeta *renderMeta, CameraMeta *cameraMeta, BVHCompact *bvhCompact,
                                  GeometryCompact *geometryCompact, MaterialCompact *materialCompact) {

    // assign a CUDA thread to every pixel by using the threadIndex
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // get window size from camera
    int width = cameraMeta->resolution.x;
    int height = cameraMeta->resolution.y;
//    float tMin = 0.00001f; // set to 0.01f when using refractive material
    // TODO: find a proper way to set minimal ray distance.
    float tMin = 0.01f; // set to 0.01f when using refractive material
    float tMax = 1e20;

    // global threadId, see richiesams blogspot
    int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) +
                   (threadIdx.y * blockDim.x) + threadIdx.x;

    curandState randState; // state of the random number generator, to prevent repetition
    curand_init(renderMeta->hashedFrame + threadId, 0, 0, &randState);

    Vec3f finalColor; // final pixel color
    finalColor = Vec3f(0.0f, 0.0f, 0.0f); // reset color to zero for every pixel
    Vec3f cameraPosition = Vec3f(cameraMeta->position.x, cameraMeta->position.y, cameraMeta->position.z);

    int i = (height - y - 1) * width + x; // pixel index in buffer
    int pixelX = x; // pixel x-coordinate on screen
    int pixelY = height - y - 1; // pixel y-coordinate on screen

    for (int s = 0; s < renderMeta->SAMPLES; s++) {
        // compute primary ray direction
        // use camera view of current frame (transformed on CPU side) to create local orthonormal basis
        Vec3f cameraView = Vec3f(cameraMeta->view.x, cameraMeta->view.y, cameraMeta->view.z);
        cameraView.normalize();
        // view is already supposed to be normalized, but normalize it explicitly just in case.
        Vec3f cameraUp = Vec3f(cameraMeta->up.x, cameraMeta->up.y, cameraMeta->up.z);
        cameraUp.normalize();
        Vec3f horizontalAxis = cross(cameraView, cameraUp);
        horizontalAxis.normalize(); // Important to normalize!
        Vec3f verticalAxis = cross(horizontalAxis, cameraView);
        verticalAxis.normalize();
        // verticalAxis is normalized by default, but normalize it explicitly just for good measure.

        Vec3f middle = cameraPosition + cameraView;
        Vec3f horizontal = horizontalAxis * tanf(cameraMeta->fov.x * 0.5f * (renderMeta->PI / 180));
        // Treating FOV as the full FOV, not half, so multiplied by 0.5
        Vec3f vertical = verticalAxis * tanf(-cameraMeta->fov.y * 0.5f * (renderMeta->PI / 180));
        // Treating FOV as the full FOV, not half, so multiplied by 0.5

        // anti-aliasing
        // calculate center of current pixel and add random number in X and Y dimension
        // based on https://github.com/peterkutz/GPUPathTracer

        float jitterValueX = curand_uniform(&randState) - 0.5f;
        float jitterValueY = curand_uniform(&randState) - 0.5f;
        float sx = (jitterValueX + pixelX) / (cameraMeta->resolution.x - 1);
        float sy = (jitterValueY + pixelY) / (cameraMeta->resolution.y - 1);

        // compute pixel on screen
        Vec3f pointOnPlaneOneUnitAwayFromEye = middle + (horizontal * ((2 * sx) - 1)) + (vertical * ((2 * sy) - 1));
        Vec3f pointOnImagePlane = cameraPosition +
                                  ((pointOnPlaneOneUnitAwayFromEye - cameraPosition) * cameraMeta->focalDistance);
        // Important for depth of field!

        // calculation of depth of field / camera aperture
        // based on https://github.com/peterkutz/GPUPathTracer

        Vec3f aperturePoint;

        if (cameraMeta->apertureRadius > 0.00001) { // the small number is an epsilon value.

            // generate random numbers for sampling a point on the aperture
            float random1 = curand_uniform(&randState);
            float random2 = curand_uniform(&randState);

            // randomly pick a point on the circular aperture
            float angle = renderMeta->TWO_PI * random1;
            float distance = cameraMeta->apertureRadius * sqrtf(random2);
            float apertureX = cos(angle) * distance;
            float apertureY = sin(angle) * distance;

            aperturePoint = cameraPosition + (horizontalAxis * apertureX) + (verticalAxis * apertureY);
        } else {
            // zero aperture
            aperturePoint = cameraPosition;
        }

        // calculate ray direction of next ray in path
        Vec3f apertureToImagePlane = pointOnImagePlane - aperturePoint;
        apertureToImagePlane.normalize(); // ray direction needs to be normalised

        // ray direction
        Vec3f directionInWorldSpace = apertureToImagePlane;
        directionInWorldSpace.normalize();

        // ray origin
        Vec3f originInWorldSpace = aperturePoint;

        Ray rayInWorldSpace(originInWorldSpace, directionInWorldSpace, tMin, tMax);
        finalColor += renderKernel(&randState, hdrEnv, bvhCompact, geometryCompact, materialCompact, rayInWorldSpace,
                                   renderMeta) * (1.0f / renderMeta->SAMPLES);
    }

    // add pixel color to accumulation buffer (accumulates all samples)
    accumulatedBuffer[i] += finalColor;

    // averaged color: divide color by the number of calculated frames so far
    Vec3f tmpColor = accumulatedBuffer[i] / renderMeta->frameNumber;

    Color outputColor;
    Vec3f color = Vec3f(clamp(tmpColor.x, 0.0f, 1.0f), clamp(tmpColor.y, 0.0f, 1.0f), clamp(tmpColor.z, 0.0f, 1.0f));

    // convert from 96-bit to 24-bit color + perform gamma correction
    outputColor.components = make_uchar4((unsigned char) (powf(color.x, 1 / 2.2f) * 255),
                                         (unsigned char) (powf(color.y, 1 / 2.2f) * 255),
                                         (unsigned char) (powf(color.z, 1 / 2.2f) * 255), 1);

    // store pixel coordinates and pixel color in OpenGL readable output buffer
    outputBuffer[i] = Vec3f(x, y, outputColor.c);
}

void Renderer::render() {
    dim3 block(16, 16, 1);
    dim3 grid(config->width / block.x, config->height / block.y, 1);
    // Configure grid and block sizes:
    int threadsPerBlock = 256;
    // Compute the number of blocks required, performing a ceiling operation to make sure there are enough:
    int fullBlocksPerGrid = ((config->width * config->height) + threadsPerBlock - 1) / threadsPerBlock;

    // copy the CPU rendering parameter to a GPU rendering parameter
    cudaMemcpy(renderMetaDevice, &renderMeta, sizeof(RenderMeta), cudaMemcpyHostToDevice);

    pathTracingKernel << < grid, block >> > (outputBuffer, accumulatedBuffer, hdrEnv,
            renderMetaDevice, cameraMetaDevice, bvhCompact, geometryCompact, materialCompact);
}

