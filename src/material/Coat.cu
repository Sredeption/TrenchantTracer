#include <material/Coat.h>

const std::string Coat::TYPE = "Coat";

__host__ __device__ Coat::Coat() : Material(COAT) {
    //hand coded test parameter
    specularColor = Vec3f(1, 1, 1);
    diffuseColor = Vec3f(0.9f, 0.3f, 0.0f);
}

__host__ Coat::Coat(const nlohmann::json &material) : Coat() {
}

__device__ Ray Coat::sample(curandState *randState, const Ray &ray, const Hit &hit, Vec3f &mask) {
    Ray nextRay = ray;// ray of next path segment
    float rouletteRandomFloat = curand_uniform(randState);
    float threshold = 0.05f;
    bool reflectFromSurface = (rouletteRandomFloat < threshold);
    //computeFresnel(make_Vec3f(n.x, n.y, n.z), incident, incidentIOR, transmittedIOR, reflectionDirection, transmissionDirection).reflectionCoefficient);

    if (reflectFromSurface) { // calculate perfectly specular reflection

        // Ray reflected from the surface. Trace a ray in the reflection direction.
        // TODO: Use Russian roulette instead of simple multipliers!
        // (Selecting between diffuse sample and no sample (absorption) in this case.)

        mask *= specularColor;
        nextRay.direction = ray.direction - hit.n * 2.0f * dot(hit.n, ray.direction);
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
        mask *= diffuseColor;
    }

    return nextRay;
}
