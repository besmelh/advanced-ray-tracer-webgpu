// control keys:
// 0 - rotate camera around scene
// 9 - increase focal length
// 8 - decrease focal length
// 7 - increase aperture
// 6 - decrease aperture



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
  var n = 3;
  var pixel_color = vec3<f32>(0,0,0);
  var stratum_size = 1.0 / f32(n); // Size of each stratum

  for (var p = 0; p <= n-1; p++) {
    for (var q = 0; q <= n-1; q++) {
    // Compute the starting corner of the stratum
      var stratum_origin = vec2<f32>(f32(p), f32(q)) * stratum_size;
      
      // Generate a random point within the stratum
      var random_offset = vec2<f32>(rnd(), rnd()) * stratum_size;
      var sample_point = stratum_origin + random_offset;
      
      // Compute sample's coordinates on the image plane
      var ui = _left + (_right - _left) * ((pixel_position.x + sample_point.x) / image_resolution.x);
      var vi = _bottom + (_top - _bottom) * ((pixel_position.y + sample_point.y) / image_resolution.y);
      
      // cur_time = 0.5 * f32(p + q) / f32(n * n);
      cur_time =  0.5 * f32(p + q) / f32(n * n);

      var ray:Ray = get_ray(camera, ui, vi);

      // find the convergence to calculate the depth of field
      var converegence_point = ray.orig + camera.focal_length * ray.dir;
      var shifted_ray:Ray = get_shifted_ray(converegence_point, ray);

      pixel_color = pixel_color + get_pixel_color(shifted_ray);
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
  camera.dir = normalize(camera.lookat-camera.origin);
  camera.aperture = get_aperture();
  camera.focal_length = get_focal_length();

  camera.w = normalize(camera.dir*-1);
  camera.u = normalize(cross(vec3<f32>(0,1,0), camera.w));
  camera.v = cross(camera.w,camera.u);

}

fn setup_scene_objects(){
// ----------------------------------------------------------------------------
// Scene definition
// ----------------------------------------------------------------------------
  // -- Sphere[0] reflective red-- 
  world_spheres[0].center=vec3<f32>(0, 0.25, 0);
  world_spheres[0].in_motion = false;
  world_spheres[0].radius= 0.25;
  world_spheres[0].material.ambient=vec3<f32>(0.7,0.0,0.0);
  world_spheres[0].material.reflectivity=f32(1);
  world_spheres[0].material.specular=vec3<f32>(0,0,0);

  // -- Sphere[1] reflective blue-- 
//   world_spheres[1].center=vec3<f32>(-0.5, 0.25, 0.5);
//   world_spheres[1].radius= 0.25;
//   world_spheres[1].material.ambient=vec3<f32>(0,0.0,0.7);
//   world_spheres[1].material.reflectivity=f32(1);
//   world_spheres[1].material.specular=vec3<f32>(0,0,0);

    // -- Sphere[2] big blue -- 
  world_spheres[1].center=vec3<f32>(0.18, 0.4, -1);
  world_spheres[1].in_motion = true;
  world_spheres[1].center_0=vec3<f32>(0.18, 0.4, -1);
  world_spheres[1].center_1=vec3<f32>(0.18, 0.9, -1);
  world_spheres[1].radius= 0.4;
  world_spheres[1].material.ambient=vec3<f32>(0,0,1);
  world_spheres[1].material.reflectivity=f32(0);
  world_spheres[1].material.specular=vec3<f32>(0,0,0);

  // -- Sphere[3] big red -- 
  world_spheres[2].center=vec3<f32>(0.18, 0.4, 0.9);
  world_spheres[2].in_motion = false;
  world_spheres[2].radius= 0.4;
  world_spheres[2].material.ambient=vec3<f32>(1,0,0);
  world_spheres[2].material.reflectivity=f32(0.8);
  world_spheres[2].material.specular=vec3<f32>(0,0,0);

  // -- cone[0] -- 
  // world_cones[0].center=vec3<f32>( 0.18,  0.0, -1);
  // world_cones[0].radius= 0.25;
  // world_cones[0].height= 0.75;
  // world_cones[0].material.ambient=vec3<f32>(0,0,1);
  // world_cones[0].material.reflectivity=f32(0.5);
  // world_cones[0].material.specular=vec3<f32>(0, 0, 0);

  // -- cube[0] -- 
  // world_cubes[0].min=vec3<f32>( -1.0,  0.0, -1.0);
  // world_cubes[0].max=vec3<f32>(  1.0,  0.9, -1.05);
  // world_cubes[0].material.ambient=vec3<f32>(0.7,0.4,0.7);

// triangles are used for the ground
  // -- Triangle[0] -- 
  world_triangles[0].a=vec3<f32>(-2.0, 0.0, -2.0);
  world_triangles[0].b=vec3<f32>(-2.0, 0.0,  2.0);
  world_triangles[0].c=vec3<f32>(2.0, 0.0, -2.0);
  world_triangles[0].uv_a=vec2<f32>(0,0);
  world_triangles[0].uv_b=vec2<f32>(1,0);
  world_triangles[0].uv_c=vec2<f32>(0,1);
  world_triangles[0].material.ambient= vec3<f32>(0.0, 0.0, 0.0);
  world_triangles[0].material.reflectivity=f32(0);
  world_triangles[0].material.specular=vec3<f32>(0,0,0);

  // -- Triangle[1] -- 
  world_triangles[1].a=vec3<f32>(-2.0, 0.0,  2.0);
  world_triangles[1].b=vec3<f32>(2.0, 0.0,  2.0);
  world_triangles[1].c=vec3<f32>( 2.0, 0.0, -2.0);
  world_triangles[1].uv_a=vec2<f32>(1,0);
  world_triangles[1].uv_b=vec2<f32>(1,1);
  world_triangles[1].uv_c=vec2<f32>(0,1);
  world_triangles[1].material.ambient=vec3<f32>(0.0, 0.0, 0.0); 
  world_triangles[1].material.reflectivity=f32(0);
  world_triangles[1].material.specular=vec3<f32>(0,0,0);
}

fn get_ray(camera:Camera, ui:f32, vj:f32)->Ray{
  var ray: Ray;
  ray.orig = camera.origin;
  ray.dir = normalize((camera.w*-1)+(camera.u*ui)+ (camera.v*vj));
  ray.t_min = 0;
  ray.t_max = 10000.0;
  return ray;
}

// getting a random ray from our camera plane to implement depth of field
// we pass in the convergence point C = O + fD
// note: randomness is currently not stratified
fn get_shifted_ray(converegence_point:vec3<f32>, orig_ray: Ray)->Ray{  
  var shifted_ray = orig_ray; //note: double check it's a deep copy not pointer refernce
  var random_disk_point = random_from_circle() * camera.aperture;
  var r_shift = vec3<f32>(random_disk_point.x, random_disk_point.y, 0.0);
  // var r_shift = vec3<f32>(0.0, 0.08, 0.0);
  shifted_ray.dir = normalize(converegence_point - (orig_ray.orig + r_shift));
  return shifted_ray;
}

// get a random point from a circular camera plane
fn random_from_circle() -> vec2<f32> {
    var p: vec2<f32>;
    loop {
        // Generate a point `p` with each component being a random number between -1.0 and 1.0
        p = 2.0 * vec2<f32>(rnd(), rnd()) - vec2<f32>(1.0, 1.0);

        // Check if the point lies within the unit disk by ensuring its distance from the origin (0,0) is less than 1
        if (dot(p, p) < 1.0) {
            break; // If the point is inside the unit disk, exit the loop
        }
    }
    return p; // Return the random point within the unit disk
}

// // function that only calculates the direct lighting and is called by `get_pixel_color`.
fn compute_direct_shading(ray: Ray, rec:HitRecord) -> vec3<f32> {
  let ambient = rec.hit_material.ambient;
  var diffuse = vec3<f32>(0.0, 0.0, 0.0);
  var specular_highlight = vec3<f32>(0.0, 0.0, 0.0);
  // var specular_highlight = rec.hit_material.specular;
  var attenuation = 1.0;

  // create a paralleogram representation of the light
  // optimize by making these global variables
  let light_c = vec3<f32>(light.position.x - 0.5, light.position.y - 0.5, light.position.z); //corner point
  let light_a = vec3<f32>(light.position.x + 0.5, 0, 0); //edge 1 vector
  let light_b = vec3<f32>(0, light.position.y + 0.5, 0); //edge 2 vector

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

      diffuse += compute_diffuse(lightDir, rec.normal);

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
          specular_highlight += compute_specular(ray.dir, lightDir, rec.normal);
        }
      }
   }

    // average and set values
    diffuse = diffuse / f32(num_shadow_samples);
    specular_highlight = specular_highlight / f32(num_shadow_samples);

    // compute the soft shadow factor
    let soft_shadow_factor = shadow_accumulator / f32(num_shadow_samples);
    attenuation = mix(1.0, 0.3, soft_shadow_factor);

    // // apply the soft shadow factor to the diffuse and specular components
    // diffuse = diffuse * soft_shadow_factor;
    // specular_highlight = specular_highlight * soft_shadow_factor;

    // combine the lighting components
    // var this_ks  = rec.hit_material.reflectivity;
    // return ambient * Ka + (diffuse * Kd  + specular_highlight * this_ks) * attenuation;
    return ambient * Ka + (diffuse * Kd  + specular_highlight * Ks) * attenuation;
}

 
// Trace ray and return the resulting contribution of this ray
fn get_pixel_color(ray: Ray) -> vec3<f32> {
  // Sample the environment map regardless of whether the ray hits an object.
  let background_texture = sample_cubemap(ray.dir);

  // count how many reflection bounces we find for this ray
  var refl_rays_count = 0;

  // preset all values in refl_rays
  for (var i: i32 =0; i < 3; i++){
    refl_rays[i] = ray;
  }

  var final_pixel_color = vec3<f32>(0,0,0);

  var rec = trace_ray(ray);
  if(!rec.hit_found) // if hit background
  {
    final_pixel_color = background_texture;
  }
  else
  {
    final_pixel_color = compute_direct_shading(ray, rec);

    var cur_ray = ray;
    var cur_rec = rec;

    // first reflective ray
    if (cur_rec.hit_material.reflectivity > 0){
      
      // my first reflective ray is a result of whatever is produced by this info
      refl_rays_count += 1;
      refl_rays[0] = cur_ray;
      refl_recs[0] = cur_rec;

      // we only want to have up to 3 reflection rays total - including the one counted above
      for (var i: i32 = 1; i < 3; i++){
        // check what the cur reflective surface points to
        cur_ray = compute_glossy_reflection_ray(cur_ray, cur_rec);
        cur_rec = trace_ray(cur_ray);
        refl_rays[i] = cur_ray;
        refl_recs[i] = cur_rec;
        refl_rays_count += 1;

        // if this ray hits nothing
        if (!cur_rec.hit_found){
          break;
        }

        // if this ray hits a non reflective surface
        if (cur_rec.hit_material.reflectivity == 0){
          break;
        } 
      }
    }
  }

  // retreive reflection textures that correspond to each of our computed reflection rays
  // if a reflection ray doesn't exist, we just give it a default values correspondingto the original ray
  refl_rays_bg[0] = background_texture;
  for (var i: i32 = 1; i < 3; i++){
    refl_rays_bg[i] = sample_cubemap(refl_rays[i].dir);
  }

  // loop through our reflection rays array and add the values to the final color
  for (var i: i32 = 1; i < refl_rays_count; i++) {
    // if the reflection ray hits another object
    // var cur_refl_rec = trace_ray(refl_recs[i]);
    if (refl_recs[i].hit_found){
      final_pixel_color += compute_direct_shading(refl_rays[i], refl_recs[i]) * refl_recs[i-1].hit_material.reflectivity;
      // final_pixel_color += get_reflection_color(refl_rays[i]) * refl_recs[i].hit_material.reflectivity;
    } 
    // if no object is hit, add reflection_texture of the background
    else {
      final_pixel_color += refl_rays_bg[i];
    }

  }
  
  return final_pixel_color;
}

// Trace reflection ray and return the resulting color contribution of this ray
fn get_reflection_color(ray: Ray) -> vec3<f32> {
  var rec = trace_ray(ray);
  
  if(!rec.hit_found) { // if hit background
    return get_background_color();
  } 
  return compute_direct_shading(ray, rec);
}

fn trace_ray(ray: Ray) -> HitRecord{
   var hitWorld_rec:HitRecord;
   var hit_found = false;
	 var closest_so_far = ray.t_max;
   //=========================
  for (var i: i32 = 0; i < world_spheres_count; i++) {
      if (world_spheres[i].in_motion){
        world_spheres[i].center = world_spheres[i].center_0 * (1.0 - cur_time) + world_spheres[i].center_1 * cur_time;
      }
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
    return light.color * specular;
}

// compute the ray of the the glossy reflection
// assumes we already checked that a reflection does indeed exist for this surface
fn compute_glossy_reflection_ray(ray:Ray, rec:HitRecord)-> Ray
{
  let r = reflect(ray.dir, rec.normal); 
  
  // reflection square side length = a, this represents the surface roughness
  let a = 1 - rec.hit_material.reflectivity;
  // selecting random points from square
  let rand_u = -1 * (a/2) + rnd() * a;
  let rand_v = -1 * (a/2) + rnd() * a;

  // find the edge vectors of the square plane
  let r_norm = normalize(r);
  let up = vec3<f32>(0.0, 1.0, 0.0); // Use a generic up vector
  let edge_u = normalize(cross(up, r_norm));
  let edge_v = cross(r_norm, edge_u);

  // note: might need to be normalized
  // r' = r + rand_u * edge_u + rand_v * edge_v
  let reflection_dir = normalize(r + rand_u * edge_u + rand_v * edge_v);

  var reflection_ray: Ray;
  reflection_ray.orig = rec.p + reflection_dir + 0.001; // Offset a bit to prevent self-intersection
  reflection_ray.dir = reflection_dir;
  reflection_ray.t_min = 0.001;
  reflection_ray.t_max = 10000.0;

  return reflection_ray;
}


fn get_checkerboard_texture_color(uv:vec2<f32>)->vec3<f32>{
    var cols=10.0;
    var rows=10.0;
    var total = floor(uv.x * cols) +
                  floor(uv.y * rows);
    if(modulo(total, 2.0) == 0.0)
    {
      // green
      return vec3<f32>(0,0.4,0);
    }
    else
    {
      // white
      return vec3<f32>(0.8);
    }
}

// returns gradient background
fn get_background_color()->vec3<f32>{
    let t = pixel_position.y / image_resolution.y;
    return t*vec3<f32>(0.2, 0.2, 0.2) + (1.0-t)*vec3<f32>(1.0, 1.0, 1.0);
}


// returns the coordinates on the texture 
// layout of faces:
//        [Top]
// [Left] [Front] [Right] [Back]
//        [Bottom]
fn get_cubemap_uv(reflection_dir: vec3<f32>)->vec2<f32>{

    var uv = vec2<f32>(0, 0);
    let dir = normalize(reflection_dir);
    let abs_dir = abs(dir);
    
    // determine which cubemap face the direction vector is pointing to.
    // x-face ********************************
    if (abs_dir.x > abs_dir.y && abs_dir.x > abs_dir.z){
      // + x, right face
      if (reflection_dir.x > 0){
        uv = vec2<f32>(-dir.z / abs_dir.x, -dir.y / abs_dir.x);
        uv = uv * 0.5 + 0.5;
        // shift to the right face's horizontal position
        uv.x = uv.x * 0.25 + 0.5; 
      } 
      // - x, left face
      else {
        uv = vec2<f32>(dir.z / abs_dir.x, -dir.y / abs_dir.x);
        uv = uv * 0.5 + 0.5;
        // shift to the left face's horizontal position
        uv.x = uv.x * 0.25;
      }
      // adjust to vertical position
      uv.y = uv.y * (1.0 / 3.0) + (1.0 / 3.0); 
    } 

    // y-face ********************************
    else if (abs_dir.y > abs_dir.x && abs_dir.y > abs_dir.z){
      // + y, top face
      if (reflection_dir.y > 0){
        uv = vec2<f32>(dir.x / abs_dir.y, dir.z / abs_dir.y);
        uv = uv * 0.5 + 0.5;
        // shift to the top face's vertical position
        uv.y = uv.y * (1.0 / 3.0);
      } 
      // - y, bottom face
      else {
        uv = vec2<f32>(dir.x / abs_dir.y, -dir.z / abs_dir.y);
        uv = uv * 0.5 + 0.5;
        // shift to the bottom face's horizontal position
        uv.y = uv.y * (1.0 / 3.0) + (2.0 / 3.0); 
      }
      // adjust to horizontal position between left and right faces
      uv.x = uv.x * 0.25 + 0.25; 
    }

    // z-face ********************************
    else if (abs_dir.z > abs_dir.x && abs_dir.z > abs_dir.y){
      // + z, front face
      if (reflection_dir.y > 0){
        uv = vec2<f32>(dir.x / abs_dir.z, -dir.y / abs_dir.z);
        uv = uv * 0.5 + 0.5;
        // shift to the front face's horizontal position
        uv.x = uv.x * 0.25 + 0.25;
      } 
      // - z, back face
      else {
        uv = vec2<f32>(-dir.x / abs_dir.z, -dir.y / abs_dir.z);
        uv = uv * 0.5 + 0.5;
        // shift to the back face's horizontal position
        uv.x = uv.x * 0.25 + 0.75; 
      }
      // adjust to the middle vertical position
      uv.y = uv.y * (1.0 / 3.0) + (1.0 / 3.0); 
    }

    // flip the y-coordinate to match the texture's origin at the top-left corner
    // uv.y = 1.0 - uv.y;
    return uv;
}

// returns color from cubemap
fn sample_cubemap(direction: vec3<f32>) -> vec3<f32> {
    let uv = get_cubemap_uv(direction);  // Get the UV coordinates from the direction
    let color = textureSample(texture1, sampler_, uv); // Sample the cubemap texture
    return color.rgb;  // Return the color from the cubemap
}

//-----------------------
// buffer functions
//-----------------------
fn update_buffers(){
  update_camera_theta();
  update_focal_length();
  update_aperture();
}

fn get_camera_theta()->f32{
  return floatBuffer[0];
}
fn get_focal_length()->f32{
  // initial value
  if (floatBuffer[1] <= 0){
    floatBuffer[1] = 1;
    // floatBuffer[1] = 0;
  }
  return floatBuffer[1];
}
fn get_aperture()->f32{
  // initial value
  if (floatBuffer[2] <= 0){
    // floatBuffer[2] = 0.01;
    floatBuffer[2] = 0;
  }
  return floatBuffer[2];
}

fn update_camera_theta(){
  if(Key == 48) // key==0   
  {
    var theta = floatBuffer[0];
    // theta = theta + 0.01;
    theta = theta + 0.05;
    if(theta>=360) 
    {
       theta = 0.0;
    }  
    floatBuffer[0] = theta;
  }
}
fn update_focal_length(){
  if(Key == 57) // key==9   
  {
    var focal_length = floatBuffer[1];
    focal_length = focal_length + 0.25;
    floatBuffer[1] = focal_length;
  }
  if(Key == 56) // key==8   
  {
    var focal_length = floatBuffer[1];
    focal_length = focal_length - 0.25;
    if(focal_length<0) 
    {
       focal_length = 0.0;
    } 
    floatBuffer[1] = focal_length;
  }
}
fn update_aperture(){
  if(Key == 55) // key==7   
  {
    var aperture = floatBuffer[2];
    aperture = aperture + 0.005;
    floatBuffer[2] = aperture;
  }
  if(Key == 54) // key==6   
  {
    var aperture = floatBuffer[2];
    aperture = aperture - 0.005;
    if(aperture<0) 
    {
       aperture = 0.0;
    } 
    floatBuffer[2] = aperture;
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
  ambient: vec3<f32>,
  specular: vec3<f32>,
  reflectivity: f32
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
  in_motion: bool,
  center_0:vec3<f32>, // center at time = 0
  center_1:vec3<f32>, // center at time = 1
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
  focal_length:f32,
  // aperture represents the side-length of the camera plane, which affects the amount of blurriness
  aperture:f32
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
var<private> Kg:f32 = 0.5; //glossy reflection

var<private> pixel_position: vec2<f32>;
var<private> image_resolution: vec2<f32>;

// world objects
var<private> world_spheres_count: i32 = 4;
var<private> world_spheres: array<Sphere, 4>;

var<private> world_triangles_count: i32 = 2;
var<private> world_triangles: array<Triangle, 2>;

var<private> world_cones_count: i32 = 1;
var<private> world_cones: array<Cone, 1>;


// world objects
var<private> world_cubes_count: i32 = 1;
var<private> world_cubes: array<Cube, 1>;

// random number generator
var<private> seed: u32 = 0;

// random number generator
var<private> cur_time: f32 = 0;

// save the reflective rays
var<private> refl_rays: array<Ray, 3>;
var<private> refl_recs: array<HitRecord, 3>;
var<private> refl_rays_bg: array<vec3<f32>, 3>; //points to texture bg