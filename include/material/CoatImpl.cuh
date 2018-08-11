#ifndef TRENCHANTTRACER_COATIMPL_H
#define TRENCHANTTRACER_COATIMPL_H

#include <material/Coat.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <geometry/Ray.h>
#include <geometry/Hit.h>

__device__ __inline__ Vec3f reflect(const Ray &ray, const Hit &hit) {
    return ray.direction - hit.n * 2.0f * dot(hit.n, ray.direction);
}

__device__ __inline__ Vec3f
refract(const Ray &ray, const Hit &hit, float refractiveIndexIncident, float refractiveIndexTransmitted) {
    // Snell's Law:
    // Copied from Photorealizer.

    float cosTheta1 = dot(hit.n, -ray.direction);

    float n1_n2 = refractiveIndexIncident / refractiveIndexTransmitted;

    float radicand = 1 - powf(n1_n2, 2) * (1 - powf(cosTheta1, 2));
    if (radicand < 0) return Vec3f(0, 0, 0); // Return value
    float cosTheta2 = sqrtf(radicand);

    if (cosTheta1 > 0) { // normal and incident are on same side of the surface.
        return ray.direction * n1_n2 + hit.n * (cosTheta1 * n1_n2 - cosTheta2);
    } else { // normal and incident are on opposite sides of the surface.
        return ray.direction * n1_n2 + hit.n * (cosTheta1 * n1_n2 + cosTheta2);
    }

}

__host__ __device__
float fresnel(const Ray &ray, const Hit &hit, float refractiveIndexIncident,
              float refractiveIndexTransmitted, const Vec3f &reflectDir,
              const Vec3f &transmitDir) {

    // First, check for total internal reflection:
    if (transmitDir.length() <= 0.12345 ||
        dot(hit.n, transmitDir) > 0) { // The length == 0 thing is how we're handling TIR right now.
        // Total internal reflection!
        return 1;
    }

    // Real Fresnel equations:
    // Copied from Photorealizer.
    float cosThetaIncident = dot(hit.n, -ray.direction);
    float cosThetaTransmitted = dot(-hit.n, transmitDir);
    float reflectionCoefficientSPolarized = powf(
            (refractiveIndexIncident * cosThetaIncident - refractiveIndexTransmitted * cosThetaTransmitted) /
            (refractiveIndexIncident * cosThetaIncident + refractiveIndexTransmitted * cosThetaTransmitted), 2);
    float reflectionCoefficientPPolarized = powf(
            (refractiveIndexIncident * cosThetaTransmitted - refractiveIndexTransmitted * cosThetaIncident) /
            (refractiveIndexIncident * cosThetaTransmitted + refractiveIndexTransmitted * cosThetaIncident), 2);
    float reflectionCoefficientUnpolarized =
            (reflectionCoefficientSPolarized + reflectionCoefficientPPolarized) / 2.0f; // Equal mix.
    return reflectionCoefficientUnpolarized;

}

// COAT material based on https://github.com/peterkutz/GPUPathTracer
// randomly select diffuse or specular reflection
// looks okay-ish but inaccurate (no Fresnel calculation yet)
__device__ __inline__ Ray coatSample(Coat *coat, curandState *randState, const Ray &ray, const Hit &hit, Vec3f &mask) {
    Ray nextRay = ray;// ray of next path segment

    Vec3f reflectDir = reflect(ray, hit);
    float rouletteRandomFloat = curand_uniform(randState);
    float nAir = 1.000293; // TODO: Generalize the medium system
    float nCoat = 1.291;
    Vec3f refractDir = refract(ray, hit, nAir, nCoat);
    float threshold = fresnel(ray, hit, nAir, nCoat, reflectDir, refractDir);

    bool reflectFromSurface = (rouletteRandomFloat < threshold);

    if (reflectFromSurface) { // calculate perfectly specular reflection

        // Ray reflected from the surface. Trace a ray in the reflection direction.
        // (Selecting between diffuse sample and no sample (absorption) in this case.)

        mask *= coat->specularColor;
        nextRay.direction = reflectDir;
        nextRay.direction.normalize();

        // offset origin next path segment to prevent self intersection
        nextRay.origin = hit.point + hit.nl * 0.001f; // scene size dependent
    } else {  // calculate perfectly diffuse reflection

        float r1 = 2 * M_PI * curand_uniform(randState);
        float r2 = curand_uniform(randState);
        float r2s = sqrtf(r2);

        // compute orthonormal coordinate frame uvw with hitpoint as origin
        Vec3f w = hit.nl;
        w.normalize();
        Vec3f u = cross((fabs(w.x) > .1 ? Vec3f(0, 1, 0) : Vec3f(1, 0, 0)), w);
        u.normalize();
        Vec3f v = cross(w, u);

        // compute cosine weighted random ray direction on hemisphere
        nextRay.direction = u * cosf(r1) * r2s + v * sinf(r1) * r2s + w * sqrtf(1 - r2);
        nextRay.direction.normalize();

        // offset origin next path segment to prevent self intersection
        nextRay.origin = hit.point + hit.nl * 0.001f;  // // scene size dependent

        // multiply mask with colour of object
        mask *= coat->diffuseColor;
    }

    return nextRay;
}

#endif //TRENCHANTTRACER_COATIMPL_H
