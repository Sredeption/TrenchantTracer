#include <core/Renderer.h>

#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <math/CutilMath.h>
#include <geometry/Ray.h>
#include <material/Coat.h>
#include <material/Diff.h>
#include <material/Metal.h>
#include <material/Spec.h>
#include <material/Refr.h>

// union struct required for mapping pixel colours to OpenGL buffer
union Color  // 4 bytes = 4 chars = 1 float
{
    float c;
    uchar4 components;
};

__device__ __inline__ Vec3f renderKernel(curandState *randState, HDRImage *hdrEnv,
                                         BVHCompact *bvhCompact, MaterialCompact *materialCompact,
                                         Ray ray, RenderMeta *renderMeta) {

    Vec3f mask = Vec3f(1.0f, 1.0f, 1.0f); // colour mask
    Vec3f accumulatedColor = Vec3f(0.0f, 0.0f, 0.0f); // accumulated colour
    Vec3f emit;

    for (int bounces = 0; bounces < 4; bounces++) {
        // iteration up to 4 bounces (instead of recursion in CPU code)


        Hit hit = ray.intersect(bvhCompact, true);

        if (hit.distance > 1e19) {
            emit = hdrEnv->sample(ray, renderMeta);
            accumulatedColor += (mask * emit);
            return accumulatedColor;
        }

        // TRIANGLES:
        emit = Vec3f(0.0, 0.0, 0);  // object emission
        accumulatedColor += (mask * emit);

        Material *material = materialCompact->materials[hit.matIndex];
        switch (material->type) {
            case COAT:
                ray = ((Coat *) material)->sample(randState, ray, hit, mask);
                break;
            case DIFF:
                ray = ((Diff *) material)->sample(randState, ray, hit, mask);
                break;
            case METAL:
                ray = ((Metal *) material)->sample(randState, ray, hit, mask);
                break;
            case SPEC:
                ray = ((Spec *) material)->sample(randState, ray, hit, mask);
                break;
            case REFR:
                ray = ((Refr *) material)->sample(randState, ray, hit, mask);
                break;
        }
    }
    return accumulatedColor;
}

__global__ void pathTracingKernel(Vec3f *outputBuffer, Vec3f *accumulatedBuffer, HDRImage *hdrEnv,
                                  RenderMeta *renderMeta, CameraMeta *cameraMeta, BVHCompact *bvhCompact,
                                  MaterialCompact *materialCompact) {

    // assign a CUDA thread to every pixel by using the threadIndex
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // get window size from camera
    int width = cameraMeta->resolution.x;
    int height = cameraMeta->resolution.y;
//    float ray_tmin = 0.00001f; // set to 0.01f when using refractive material
    // TODO: find a proper way to set minimal ray distance.
    float ray_tmin = 0.01f; // set to 0.01f when using refractive material
    float ray_tmax = 1e20;

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

    Vec3f camDir = Vec3f(0, -0.042612f, -1);
    camDir.normalize();
    Vec3f cx = Vec3f(width * .5135f / height, 0.0f, 0.0f);  // ray direction offset along X-axis
    Vec3f cy = (cross(cx, camDir)).normalize() * .5135f; // ray dir offset along Y-axis, .5135 is FOV angle

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
        Vec3f pointOnImagePlane =
                cameraPosition + ((pointOnPlaneOneUnitAwayFromEye - cameraPosition) * cameraMeta->focalDistance);
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

        Ray rayInWorldSpace(originInWorldSpace, directionInWorldSpace, ray_tmin, ray_tmax);
        finalColor += renderKernel(&randState, hdrEnv, bvhCompact, materialCompact, rayInWorldSpace, renderMeta) *
                      (1.0f / renderMeta->SAMPLES);
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
            renderMetaDevice, cameraMetaDevice, bvhCompact, materialCompact);
}

