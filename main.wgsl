
@fragment
fn main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
  // global variable
  image_resolution = Resolution.xy;
  pixel_position = vec2<f32>(position.x, image_resolution.y - position.y);

  // for the random number generator
  let v= u32(10); //any value
  seed = tea(u32(position.x + position.y * Resolution.x), v);

// update the camera theta to rotate when '0' key is pressed
  if(pixel_position.x <= 1.0 && pixel_position.y <= 1.0)
  {   
      update_buffers();
  }

  // setup scene
  let _top = 0.88; //how zoomed in/out the image is
  let _right = (image_resolution.x/image_resolution.y)*_top;
  let _left = _right*-1;
  let _bottom = _top*-1;

  setup_camera();
  setup_light();
  setup_scene_objects();


  // perform startified supersampling for antialiasing
  var n = 4;
  var pixel_color = vec3<f32>(0,0,0);
  for (var p = 0; p <= n-1; p++) {
    for (var q = 0; q <= n-1; q++) {
      var r = rnd();
      var ui = _left + (_right - _left) * ((pixel_position.x + (f32(p) + r)/f32(n)) / image_resolution.x);
      var vi = _bottom + (_top - _bottom) * ((pixel_position.y + (f32(q) + r)/f32(n)) / image_resolution.y);
      var ray:Ray = get_ray(camera,ui,vi);
      pixel_color = pixel_color + get_pixel_color(ray);
      // pixel_color = pixel_color + vec3<f32>(1,0,0);
    }
  }
  pixel_color = pixel_color / f32(n*n);   
  
  return vec4<f32>(pixel_color,1.0);
}

fn setup_light()
{
  light.position = vec3<f32>(3.0, 3.0, 1.0);
  light.color = vec3<f32>(1.0,1.0,1.0);
}

fn setup_camera(){
  // Rotate the camera around
  var look_from = vec3<f32>(0.0, 1.0, 1.0);
  var theta = get_camera_theta();
  var rot = mat3x3<f32>(cos(theta), 0., -sin(theta),
                   0., 1., 0.,
                -sin(theta), 0., cos(theta));
      
  camera.origin = rot * look_from;
  camera.lookat =  vec3<f32>(0.0, 0.0, 0.0);
  camera.dir = normalize(camera.lookat - camera.origin);
  camera.aperture = 0.5;
  camera.focal_length = -2;

  camera.w = normalize(camera.dir * -1);
  camera.u = normalize(cross(vec3<f32>(0,1,0), camera.w));
  camera.v = cross(camera.w,camera.u);

}

fn setup_scene_objects(){
// ----------------------------------------------------------------------------
// Scene definition
// ----------------------------------------------------------------------------
  // -- Sphere[0] -- 
  world_spheres[0].center=vec3<f32>(-0.125, 0.25, 0.35);
  world_spheres[0].radius= 0.25;
  world_spheres[0].material.ambient=vec3<f32>(0.7,0.0,0.0);

  // -- cone[0] -- 
  world_cones[0].center=vec3<f32>( 0.18,  0.0, -0.12);
  world_cones[0].radius= 0.25;
  world_cones[0].height= 0.75;
  world_cones[0].material.ambient=vec3<f32>(0.0,0.4,0.7);

  // -- cube[0] -- 
  //world_cubes[0].min=vec3<f32>( -1.0,  0.0, -1.0);
  //world_cubes[0].max=vec3<f32>(  1.0,  0.9, -1.05);
  //world_cubes[0].material.ambient=vec3<f32>(0.7,0.4,0.7);

// triangles are used for the ground
  // -- Triangle[0] -- 
  world_triangles[0].a=vec3<f32>(-2.0, 0.0, -2.0);
  world_triangles[0].b=vec3<f32>(-2.0, 0.0,  2.0);
  world_triangles[0].c=vec3<f32>(2.0, 0.0, -2.0);
  world_triangles[0].uv_a=vec2<f32>(0,0);
  world_triangles[0].uv_b=vec2<f32>(1,0);
  world_triangles[0].uv_c=vec2<f32>(0,1);
  world_triangles[0].material.ambient= vec3<f32>(0.0, 0.0, 0.0);

  // -- Triangle[1] -- 
  world_triangles[1].a=vec3<f32>(-2.0, 0.0,  2.0);
  world_triangles[1].b=vec3<f32>(2.0, 0.0,  2.0);
  world_triangles[1].c=vec3<f32>( 2.0, 0.0, -2.0);
  world_triangles[1].uv_a=vec2<f32>(1,0);
  world_triangles[1].uv_b=vec2<f32>(1,1);
  world_triangles[1].uv_c=vec2<f32>(0,1);
  world_triangles[1].material.ambient=vec3<f32>(0.0, 0.0, 0.0); 
}

fn get_ray(camera:Camera,ui:f32,vj:f32)->Ray{
  var ray: Ray;
  ray.orig = camera.origin;
  ray.dir = normalize((camera.w*-1) + (camera.u*ui) + (camera.v*vj));
  ray.t_min = 0;
  ray.t_max = 10000.0;
  return ray;
}

// getting a random ray from our camera plane to implement depth of field
fn get_ray_dof(camera:Camera, ui:f32, vj:f32)->Ray{
  // var primary_ray: Ray;
  // primary_ray.orig = camera.origin;
  // primary_ray.dir = normalize((camera.w * -1) + (camera.u * ui) + (camera.v * vj));
  // primary_ray.t_min = 0;
  // primary_ray.t_max = 10000.0;

  
  var secondary_ray: Ray;

  // shift the camera origin to be somewhere random within the camera plane
  var offset = vec3<f32>(rnd() / camera.aperture, rnd() / camera.aperture, 0.0);
  vec3<f32> shifted_cam_origin =  camera.origin - vec3<f32>(camera.aperture/2, camera.aperture/2, 0.0) + offset;
  vec3<f32> shifted_cam_dir =  normalize(camera.lookat - shifted_cam_origin);

  secondary_ray.orig = shifted_cam_origin;
  primary_ray.dir = normalize((normalize(shifted_cam_dir * -1) * -1) + (camera.u * ui) + (camera.v * vj));
  primary_ray.t_min = 0;
  primary_ray.t_max = 10000.0;





  // var camera_shifted: Camera; //note: this may a shallow copy refernce
  // // shift the camera origin to be somewhere random within the camera plane
  // var offset = vec3<f32>(rnd() / camera.aperture, rnd() / camera.aperture, 0.0);
  // camera_shifted.origin = camera.origin - vec3<f32>(camera.aperture/2, camera.aperture/2, 0.0) + offset;
  
  // // other attributes
  // camera_shifted.aperture = camera.aperture;
  // camera_shifted.u = camera.u;
  // camera_shifted.v = camera.v;
  // camera_shifted.w = 

  // var secondary_ray: Ray;
  // secondary_ray.orig = camera_shifted.origin;
  // vec3<f32> shifted_cam_origin =  camera.origin - vec3<f32>(camera.aperture/2, camera.aperture/2, 0.0) + offset;
  // vec3<f32> shifted_cam_dir =  normalize(camera.lookat - shifted_cam_origin);

  // // shift the camera origin to be somewhere random within the camera plane
  // var offset = vec3<f32>(rnd() / camera.aperture, rnd() / camera.aperture, 0.0);
  // set origin to be in center of camera plane
  // secondary_ray.orig = camera.origin - vec3<f32>(camera.aperture/2, camera.aperture/2, 0.0) + offset;
  // camera.w = normalize(camera.dir * -1);
  // primary_ray.dir = normalize((normalize(camera.dir * -1) * -1) + (camera.u * ui) + (camera.v * vj));

  return ray;
}

fn compute_shading(ray: Ray, rec:HitRecord)-> vec3<f32>{
    // ambient
    let ambient = rec.hit_material.ambient;

    // diffuse
    var specular    = vec3<f32>(0,0,0);
    var attenuation = 1.0;

    // create a paralleogram representation of the light
    let light_c = vec3<f32>(light.position.x - 0.5, light.position.y - 0.5, light.position.z); //corner point
    let light_a = vec3<f32>(light.position.x + 0.5, 0, 0); //edge 1 vector
    let light_b = vec3<f32>(0, light.position.y + 0.5, 0); //edge 2 vector

    // initialize diffuse and specular
    var diffuse = vec3<f32>(0,0,0);

    // number of samples for soft shadows
    let num_shadow_samples: u32 = 16; // 4 * 4
    var shadow_accumulator: f32 = 0.0;

    for (var i: u32 = 0; i < num_shadow_samples; i++) {
      // choose random point on paralleogram light area
      let random_a = rnd();
      let random_b = rnd();
      let light_r = light_c + f32(random_a) * light_a + f32(random_b) * light_b;

      var lightDir  = light_r - rec.p;
      let lightDistance = length(lightDir);
      lightDir = normalize(lightDir);
      diffuse = diffuse + compute_diffuse(lightDir, rec.normal);

      // Tracing shadow ray only if the light is visible from the surface
      if(dot(rec.normal, lightDir) > 0.0) {
        var shadow_ray: Ray;
        shadow_ray.orig =  rec.p;
        shadow_ray.dir = lightDir;
        shadow_ray.t_min = 0.001;
        shadow_ray.t_max = lightDistance - shadow_ray.t_min;
        var shadow_rec = trace_ray(shadow_ray);
        if (shadow_rec.hit_found) { 
          // Accumulate shadow factor if occlusion is found
          shadow_accumulator += 1.0;
        } else {
          specular = specular + compute_specular(ray.dir, lightDir, rec.normal);
        }
      }
    }

    // average and set values
    diffuse = diffuse / f32(num_shadow_samples);
    specular = specular / f32(num_shadow_samples);

    // Use the soft shadow factor to attenuate the light contribution
    let soft_shadow_factor = shadow_accumulator / f32(num_shadow_samples);
    attenuation = mix(1.0, 0.3, soft_shadow_factor);

   return ambient* Ka + (diffuse*Kd + specular*Ks)* attenuation;
  // return vec3<f32>(1,0,0); 
 }
 
// Trace ray and return the resulting contribution of this ray
fn get_pixel_color(ray: Ray) -> vec3<f32> {
  var final_pixel_color = vec3<f32>(0,0,0);
  var rec = trace_ray(ray);
  if(!rec.hit_found) // if hit background
  {
     final_pixel_color = get_background_color();
  }
  else
  {
     final_pixel_color = compute_shading(ray,rec);
  }
  return final_pixel_color;
}

fn trace_ray(ray: Ray) -> HitRecord{
   var hitWorld_rec:HitRecord;
   var hit_found = false;
	 var closest_so_far = ray.t_max;
   //=========================
  for (var i: i32 = 0; i < world_spheres_count; i++) {
      let temp_rec:HitRecord = sphere_intersection_test(ray,world_spheres[i]);
      if(temp_rec.hit_found){
        hit_found = true;
        if(closest_so_far> temp_rec.t){
          closest_so_far = temp_rec.t;
          hitWorld_rec = temp_rec;
            
        }
      }
  }
  //============================
  for (var i: i32 = 0; i < world_cones_count; i++) {
      let temp_rec:HitRecord = cone_intersection_test(ray,world_cones[i]);
      if(temp_rec.hit_found){
        hit_found = true;
        if(closest_so_far> temp_rec.t){
          closest_so_far = temp_rec.t;
          hitWorld_rec = temp_rec;
            
        }
      }
  }
  //============================
  /*for (var i: i32 = 0; i < world_cubes_count; i++) {
      let temp_rec:HitRecord = cube_intersection_test(ray,world_cubes[i]);
      if(temp_rec.hit_found){
        hit_found = true;
        if(closest_so_far> temp_rec.t){
          closest_so_far = temp_rec.t;
          hitWorld_rec = temp_rec;
            
        }
      }
  }*/
  //============================
  for (var i: i32 = 0; i < world_triangles_count; i++) {
      let temp_rec:HitRecord = triangle_intersection_test(ray,world_triangles[i]);
      if(temp_rec.hit_found){
        hit_found = true;
        if(closest_so_far> temp_rec.t){
            closest_so_far = temp_rec.t;
            hitWorld_rec = temp_rec;
            
        }
      }
  }
   //=========================
   hitWorld_rec.hit_found = hit_found;
   return hitWorld_rec;
}

fn sphere_intersection_test(ray: Ray, sphere:Sphere)-> HitRecord {
	var hit_rec:HitRecord;
  hit_rec.hit_found = false;
  //==================
  let oc = ray.orig - sphere.center;
	let a = dot(ray.dir, ray.dir);
	let half_b = dot(oc, ray.dir);
	let c = dot(oc, oc) - sphere.radius * sphere.radius;
	let discriminant = half_b * half_b - a * c;
	if (discriminant < 0) {
		 return hit_rec;
	}

  let sqrtd = sqrt(discriminant);
  // Find the nearest root that lies in the acceptable range.
  var root = (-half_b - sqrtd) / a;
  if ((root < ray.t_min) || (ray.t_max < root)) {
      root = (-half_b + sqrtd) / a;
      if (root < ray.t_min || ray.t_max < root){
          return hit_rec;
      }
              
  }
  hit_rec.hit_found = true;
  hit_rec.hit_material = sphere.material;
  hit_rec.t = root;
  hit_rec.p = get_hit_point(ray,hit_rec.t);
  hit_rec.normal = normalize((hit_rec.p - sphere.center) / sphere.radius);
  if(dot(ray.dir, hit_rec.normal)>=0){
      hit_rec.normal= hit_rec.normal*-1;
  }
  return hit_rec; 
}
fn triangle_intersection_test(ray: Ray, triangle:Triangle)-> HitRecord {
	var hit_rec:HitRecord;
  hit_rec.hit_found = false;
  //==================
  
	var e1 = triangle.b - triangle.a;
  var e2 = triangle.c - triangle.a;
  var q = cross(ray.dir, e2);
  let a = dot(e1, q);
    
  // No hit found so far
  if (a < ray.t_min) {
        return hit_rec;
  }
    
  var f = 1.0 / a;
  var s = ray.orig - triangle.a;
  var u = f * dot(s, q);
    
  if (u < 0.0 || u > 1.0) {
        return hit_rec;
  }
    
  var rt = cross(s, e1);
  var v = f * dot(ray.dir, rt);
    
  if (v < 0.0 || (u + v) > 1.0) {
        return hit_rec;
  }
    
  var w = (1.0 - u - v);

  // Hit found
  hit_rec.hit_found = true;
  hit_rec.t = f * dot(e2, rt);
  hit_rec.p = get_hit_point(ray,hit_rec.t);
  hit_rec.normal = normalize(cross(e1, e2));
  var uv = u * triangle.uv_b + v * triangle.uv_c + w * triangle.uv_a;
  var material= triangle.material;
  material.ambient = get_checkerboard_texture_color(uv);
  hit_rec.hit_material = material;
  return hit_rec; 
}
fn cube_intersection_test(ray: Ray, cube:Cube)-> HitRecord {
  var hit_rec:HitRecord;
  hit_rec.hit_found = false;
  //==================
  let xn = vec3<f32>(1.0, 0.0, 0.0);
	let yn = vec3<f32>(0.0, 1.0, 0.0);
	let zn = vec3<f32>(0.0, 0.0, 1.0);

  // x -> yz-plane
  var tmin = (cube.min.x - ray.orig.x) / ray.dir.x; 
  var tmax = (cube.max.x - ray.orig.x) / ray.dir.x; 
  var normal = xn;
    if (tmin > tmax) 
    {
      //swap(tmin, tmax); 
      let temp=tmin;
      tmin=tmax;
      tmax=temp;
    }

    // y -> xz-plane
    var tymin = (cube.min.y - ray.orig.y) / ray.dir.y; 
    var tymax = (cube.max.y - ray.orig.y) / ray.dir.y; 
 
    if (tymin > tymax) 
    {
      //swap(tymin, tymax); 
      let temp=tymin;
      tymin=tymax;
      tymax=temp;
    }
 
    if ((tmin > tymax) || (tymin > tmax)) 
    {
       return hit_rec; //No hit;
    }

    if (tymin > tmin) 
    { 
      tmin = tymin;
      normal = yn;  
    }
 
    if (tymax < tmax) 
    {
      tmax = tymax;
    } 
 
    // z -> xy-plane
    var tzmin = (cube.min.z - ray.orig.z) / ray.dir.z; 
    var tzmax = (cube.max.z - ray.orig.z) / ray.dir.z; 
 
    if (tzmin > tzmax) 
    {
      //swap(tzmin, tzmax);
      let temp=tzmin;
      tzmin=tzmax;
      tzmax=temp;
    } 
 
    if ((tmin > tzmax) || (tzmin > tmax)) 
    {
      return hit_rec; //No hit;
    }
 
    if (tzmin > tmin) 
    {
      tmin = tzmin; 
      normal = zn;
    }
 
    if (tzmax < tmax) 
    {
      tmax = tzmax; 
    }
 
    // Hit found
    hit_rec.hit_found = true;
    hit_rec.t = tmin;
    hit_rec.p =  get_hit_point(ray,tmin);
    hit_rec.hit_material = cube.material;
	  hit_rec.normal = normalize(normal);
    return hit_rec;
}
fn cone_intersection_test(ray: Ray, cone:Cone)-> HitRecord {
  var hit_rec:HitRecord;
  hit_rec.hit_found = false;
  //==================
  
  let oc = ray.orig - cone.center;
  let d = cone.height - ray.orig.y + cone.center.y;;

  let ratio  = (cone.radius / cone.height) * (cone.radius / cone.height);
  
  let a = (ray.dir.x * ray.dir.x) + (ray.dir.z * ray.dir.z) - (ratio *(ray.dir.y * ray.dir.y));
  let b = (2.0*oc.x*ray.dir.x) + (2.0*oc.z*ray.dir.z) + (2.0*ratio *d*ray.dir.y);
  let c = (oc.x*oc.x) + (oc.z*oc.z) - (ratio*(d*d));
  
  let delta = b*b - 4*(a*c);
	if(abs(delta) <= 0.0) 
  {
    return hit_rec;   // No hit 
  }
  
  let t1 = (-b - sqrt(delta))/(2.0*a);
  let t2 = (-b + sqrt(delta))/(2.0*a);

  var t = t1;
  if (t1>t2 || t1<0.0) 
  {
    t = t2;
  }
  let y = ray.orig.y + t*ray.dir.y;
  if (!((y > cone.center.y) && (y < cone.center.y + cone.height)) )
  {
    return hit_rec;   // No hit 
  }
  // Hit found
  hit_rec.hit_found = true;
  hit_rec.t = t;
  hit_rec.p =  get_hit_point(ray,t);

  let r = sqrt((hit_rec.p.x-cone.center.x)*(hit_rec.p.x-cone.center.x) + (hit_rec.p.z-cone.center.z)*(hit_rec.p.z-cone.center.z));
  hit_rec.normal = normalize(vec3<f32>(hit_rec.p.x-cone.center.x, r*(cone.radius/cone.height), hit_rec.p.z-cone.center.z));
  hit_rec.hit_material = cone.material;
  return hit_rec; 
}
fn get_hit_point(ray: Ray, t:f32)-> vec3<f32>{
    return ray.orig + ray.dir*t;
}
//-----------------------
// shading functions
//-----------------------
fn compute_diffuse(lightDir:vec3<f32>, normal:vec3<f32>)-> vec3<f32>
{
  //Intensity of the diffuse light.
  var ndotL = max(dot(normal, lightDir), 0.0);
  return  light.color*ndotL;
}
fn compute_specular(viewDir:vec3<f32>, lightDir:vec3<f32>, normal:vec3<f32>)-> vec3<f32>
{
    let phong_exponent=32.0;
    // Specular
    let        V                   = normalize(-viewDir);
    let        R                   = reflect(-lightDir, normal);
    let      specular            =  pow(max(dot(V, R), 0.0), phong_exponent);
    return light.color*specular;
}
fn get_checkerboard_texture_color(uv:vec2<f32>)->vec3<f32>{
    var cols=10.0;
    var rows=10.0;
    var total = floor(uv.x * cols) +
                  floor(uv.y * rows);
    if(modulo(total, 2.0) == 0.0)
    {
      return vec3<f32>(0,0.4,0);
    }
    else
    {
      return vec3<f32>(0.8);
    }
}

// returns gradient background
fn get_background_color()->vec3<f32>{
    let t = pixel_position.y / image_resolution.y;
    return t*vec3<f32>(0.2, 0.2, 0.2) + (1.0-t)*vec3<f32>(1.0, 1.0, 1.0);
}
//-----------------------
// buffer functions
//-----------------------
fn update_buffers(){
 update_camera_theta();
}

fn get_camera_theta()->f32{
  return floatBuffer[0];
}
fn update_camera_theta(){
  if(Key == 48) // key==0   
  {
    var theta = floatBuffer[0];
    theta = theta + 0.01;
    if(theta>=360) 
    {
       theta = 0.0;
    }  
    floatBuffer[0] = theta;
  }
}
//-----------------------
// math functions
//-----------------------
fn modulo(x: f32,y:f32)->f32{
  return x - (y * floor(x/y)); 
}
// ----------------------------------------------------------------------------
// struct
// ----------------------------------------------------------------------------
struct Material{
  ambient:vec3<f32>
}
struct Ray {
  orig: vec3<f32>,
  dir: vec3<f32>,
  t_min:f32,
  t_max:f32
}
struct HitRecord {
    p:vec3<f32>,
    normal:vec3<f32>,
    t:f32,
    hit_material:Material,
    hit_found:bool  
}
struct Sphere
{
  center:vec3<f32>,
  radius:f32,
  material:Material
}
struct Triangle 
{
    a:vec3<f32>,
    b:vec3<f32>,
    c:vec3<f32>,
    uv_a:vec2<f32>,
    uv_b:vec2<f32>,
    uv_c:vec2<f32>,
    material:Material
}
struct Cube
{
  min:vec3<f32>,
  max:vec3<f32>,
  material:Material
} 
struct Cone
{
  center:vec3<f32>, // base center
  radius:f32,
  height:f32,
  material:Material
} 

struct Camera
{
  origin:vec3<f32>,
  w:vec3<f32>, //lookout to origin
  u:vec3<f32>, //camera's right direction
  v:vec3<f32>, //camera's up direction
  lookat:vec3<f32>,
  dir:vec3<f32>, //origin to lookout
  // aperture represents the side-length of the camera plane, which affects the amount of blurriness
  aperture:f32,
  focal_length:f32,
}

struct Light
{
   position:vec3<f32>,
   color:vec3<f32>
}

// ----------------------------------------------------------------------------
// Random number generator
// ----------------------------------------------------------------------------
fn tea(val0:u32, val1:u32)->u32{
// "GPU Random Numbers via the Tiny Encryption Algorithm"
  var v0 = val0;
  var v1 = val1;
  var s0 = u32(0);
  for (var n: i32 = 0; n < 16; n++) {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
  }

  return v0;
}
fn rnd()->f32{
  // Generate a random float in [0, 1) 
  return (f32(lcg()) / f32(0x01000000));
}
fn lcg()->u32{
// Generate a random unsigned int in [0, 2^24) given the previous RNG state
// using the Numerical Recipes linear congruential generator
  let LCG_A = 1664525u;
  let LCG_C = 1013904223u;
  seed       = (LCG_A * seed + LCG_C);
  return seed & 0x00FFFFFF;
}

// ----------------------------------------------------------------------------
// global variables
// ----------------------------------------------------------------------------
var<private> camera: Camera;
var<private> light: Light;
var<private> Ka:f32 = 0.4;
var<private> Kd:f32 = 0.4; 
var<private> Ks:f32 = 0.2; 

var<private> pixel_position: vec2<f32>;
var<private> image_resolution: vec2<f32>;

// world objectss
var<private> world_spheres_count: i32 = 1;
var<private> world_spheres: array<Sphere, 1>;

var<private> world_triangles_count: i32 = 2;
var<private> world_triangles: array<Triangle, 2>;

var<private> world_cones_count: i32 = 1;
var<private> world_cones: array<Cone, 1>;


// world objects
var<private> world_cubes_count: i32 = 1;
var<private> world_cubes: array<Cube, 1>;

// random number generator
var<private> seed: u32 = 0;
