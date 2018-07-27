#include <core/Renderer.h>

#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <math/CutilMath.h>
#include <geometry/Ray.h>

// union struct required for mapping pixel colours to OpenGL buffer
union Color  // 4 bytes = 4 chars = 1 float
{
    float c;
    uchar4 components;
};

enum Refl_t {
    DIFF, METAL, SPEC, REFR, COAT
};  // material types

__device__ Vec3f renderKernel(curandState *randState, HDRImage *hdrEnv,
                              BVHCompact *bvhCompact, Ray ray, RenderMeta *renderMeta) {

    Vec3f mask = Vec3f(1.0f, 1.0f, 1.0f); // colour mask
    Vec3f accumulatedColor = Vec3f(0.0f, 0.0f, 0.0f); // accumulated colour
    Vec3f direct = Vec3f(0, 0, 0);
    Vec3f emit;
    Vec3f n; // normal
    Vec3f nl; // oriented normal
    Vec3f hitpoint; // intersection point
    Vec3f trinormal;
    Refl_t refltype;
    Vec3f objcol;
    Vec3f nextdir; // ray direction of next path segment

    for (int bounces = 0; bounces < 4; bounces++) {
        // iteration up to 4 bounces (instead of recursion in CPU code)

        float hitDistance = 1e20;

        Hit hit = ray.intersect(bvhCompact, false);
        hitDistance = hit.distance;
        trinormal = hit.noraml;

        if (hitDistance > 1e19) {
            emit = hdrEnv->sample(ray, renderMeta);
            accumulatedColor += (mask * emit);
            return accumulatedColor;
        }

        // TRIANGLES:
        hitpoint = ray.origin + ray.direction * hitDistance; // intersection point

        n = trinormal;
        n.normalize();
        nl = dot(n, ray.direction) < 0 ? n : n * -1;  // correctly oriented normal
        Vec3f colour = Vec3f(0.9f, 0.3f, 0.0f); // hardcoded triangle colour  .9f, 0.3f, 0.0f
        refltype = COAT; // objectmaterial
        objcol = colour;
        emit = Vec3f(0.0, 0.0, 0);  // object emission
        accumulatedColor += (mask * emit);


        // COAT material based on https://github.com/peterkutz/GPUPathTracer
        // randomly select diffuse or specular reflection
        // looks okay-ish but inaccurate (no Fresnel calculation yet)
        if (refltype == COAT) {

            float rouletteRandomFloat = curand_uniform(randState);
            float threshold = 0.05f;
            Vec3f specularColor = Vec3f(1, 1, 1);  // hard-coded
            bool reflectFromSurface = (rouletteRandomFloat <
                                       threshold); //computeFresnel(make_Vec3f(n.x, n.y, n.z), incident, incidentIOR, transmittedIOR, reflectionDirection, transmissionDirection).reflectionCoefficient);

            if (reflectFromSurface) { // calculate perfectly specular reflection

                // Ray reflected from the surface. Trace a ray in the reflection direction.
                // TODO: Use Russian roulette instead of simple multipliers!
                // (Selecting between diffuse sample and no sample (absorption) in this case.)

                mask *= specularColor;
                nextdir = ray.direction - n * 2.0f * dot(n, ray.direction);
                nextdir.normalize();

                // offset origin next path segment to prevent self intersection
                hitpoint += nl * 0.001f; // scene size dependent
            } else {  // calculate perfectly diffuse reflection

                float r1 = 2 * M_PI * curand_uniform(randState);
                float r2 = curand_uniform(randState);
                float r2s = sqrtf(r2);

                // compute orthonormal coordinate frame uvw with hitpoint as origin
                Vec3f w = nl;
                w.normalize();
                Vec3f u = cross((fabs(w.x) > .1 ? Vec3f(0, 1, 0) : Vec3f(1, 0, 0)), w);
                u.normalize();
                Vec3f v = cross(w, u);

                // compute cosine weighted random ray direction on hemisphere
                nextdir = u * cosf(r1) * r2s + v * sinf(r1) * r2s + w * sqrtf(1 - r2);
                nextdir.normalize();

                // offset origin next path segment to prevent self intersection
                hitpoint += nl * 0.001f;  // // scene size dependent

                // multiply mask with colour of object
                mask *= objcol;
            }
        } // end COAT

        // set up origin and direction of next path segment
        ray.origin = hitpoint;
        ray.direction = nextdir;
    }
    return accumulatedColor;
}

__global__ void pathTracingKernel(Vec3f *outputBuffer, Vec3f *accumulatedBuffer, HDRImage *hdrEnv,
                                  RenderMeta *renderMeta, CameraMeta *cameraMeta, BVHCompact *bvhCompact) {

    // assign a CUDA thread to every pixel by using the threadIndex
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // get window size from camera
    int width = cameraMeta->resolution.x;
    int height = cameraMeta->resolution.y;
    float ray_tmin = 0.00001f; // set to 0.01f when using refractive material
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
        finalColor += renderKernel(&randState, hdrEnv, bvhCompact, rayInWorldSpace, renderMeta) *
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

    pathTracingKernel<<<grid, block>>>(outputBuffer, accumulatedBuffer, hdrEnv,
            renderMetaDevice, cameraMetaDevice, bvhCompact);
}

