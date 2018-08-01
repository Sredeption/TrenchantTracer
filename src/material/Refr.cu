#include <material/Refr.h>

const std::string Refr::TYPE = "Refr";

__host__ __device__ Refr::Refr() : Material(REFR) {

}

__host__ Refr::Refr(const nlohmann::json &material) : Refr() {

}

__host__ U32 Refr::size() const {
    return sizeof(Refr);
}

__device__ Ray Refr::sample(curandState *randState, const Ray &ray, const Hit &hit, Vec3f &mask) {
    Ray nextRay = ray;// ray of next path segment
    bool into = dot(hit.n, hit.nl) > 0; // is ray entering or leaving refractive material?
    float nc = 1.0f;  // Index of Refraction air
    float nt = 1.4f;  // Index of Refraction glass/water
    float nnt = into ? nc / nt : nt / nc;  // IOR ratio of refractive materials
    float ddn = dot(ray.direction, hit.nl);
    float cos2t = 1.0f - nnt * nnt * (1.f - ddn * ddn);

    if (cos2t < 0.0f) {
        // total internal reflection
        nextRay.direction = ray.direction - hit.n * 2.0f * dot(hit.n, ray.direction);
        nextRay.direction.normalize();

        // offset origin next path segment to prevent self intersection
        nextRay.origin = hit.point + hit.nl * 0.001f; // scene size dependent
    } else {
        // cos2t > 0
        // compute direction of transmission ray
        Vec3f tdir = ray.direction * nnt;
        tdir -= hit.n * ((into ? 1 : -1) * (ddn * nnt + sqrtf(cos2t)));
        tdir.normalize();

        float R0 = (nt - nc) * (nt - nc) / (nt + nc) * (nt + nc);
        float c = 1.f - (into ? -ddn : dot(tdir, hit.n));
        float Re = R0 + (1.f - R0) * c * c * c * c * c;
        float Tr = 1 - Re; // Transmission
        float P = .25f + .5f * Re;
        float RP = Re / P;
        float TP = Tr / (1.f - P);

        // randomly choose reflection or transmission ray
        if (curand_uniform(randState) < 0.2) {
            // reflection ray
            mask *= RP;
            nextRay.direction = ray.direction - hit.n * 2.0f * dot(hit.n, ray.direction);
            nextRay.direction.normalize();

            nextRay.origin = hit.point + hit.nl * 0.001f; // scene size dependent
        } else {
            // transmission ray
            mask *= TP;
            nextRay.direction = tdir;
            nextRay.direction.normalize();

            nextRay.origin = hit.point + hit.nl * 0.001f; // epsilon must be small to avoid artefacts
        }
    }
    return nextRay;
}
