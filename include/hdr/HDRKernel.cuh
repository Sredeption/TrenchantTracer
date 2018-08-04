#ifndef TRENCHANTTRACER_HDRKERNEL_CUH_H
#define TRENCHANTTRACER_HDRKERNEL_CUH_H

#include <hdr/HDRImage.h>

__device__ __inline__ Vec3f hdrSample(const HDRImage *hdrImage, const Ray &ray, RenderMeta *renderMeta) {
    // if ray misses scene, return sky
    // HDR environment map code based on Syntopia "Path tracing 3D fractals"
    // http://blog.hvidtfeldts.net/index.php/2015/01/path-tracing-3d-fractals/
    // https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Include/IBL-Pathtracer.frag
    // GLSL code:
    // vec3 equirectangularMap(sampler2D sampler, vec3 dir) {
    //		dir = normalize(dir);
    //		vec2 longlat = vec2(atan(dir.y, dir.x) + RotateMap, acos(dir.z));
    //		return texture2D(sampler, longlat / vec2(2.0*PI, PI)).xyz; }

    // Convert (normalized) dir to spherical coordinates.
    float longlatX = atan2f(ray.direction.x, ray.direction.z);
    // Y is up, swap x for y and z for x
    longlatX = longlatX < 0.f ? longlatX + renderMeta->TWO_PI : longlatX;
    // wrap around full circle if negative
    float longlatY = acosf(ray.direction.y);
    // add RotateMap at some point, see Fragmentarium

    // map theta and phi to u and v texture coordinates in [0,1] x [0,1] range
    float offsetY = 0.5f;
    float u = longlatX / renderMeta->TWO_PI; // +offsetY;
    float v = longlatY / renderMeta->PI;

    // map u, v to integer coordinates
    auto u2 = (int) (u * hdrImage->width);
    auto v2 = (int) (v * hdrImage->height);

    // compute the texture index in the HDR map
    int hdrTextureIdx = u2 + v2 * hdrImage->width;

    float4 hdrColor = tex1Dfetch<float4>(hdrImage->hdrTexture, hdrTextureIdx);  // fetch from texture

    Vec3f hdrColor3 = Vec3f(hdrColor.x, hdrColor.y, hdrColor.z);

    return hdrColor3 * 2.0f;
}

#endif //TRENCHANTTRACER_HDRKERNEL_CUH_H
