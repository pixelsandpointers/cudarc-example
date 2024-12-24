use cudarc::driver::{CudaDevice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

const SRC: &str = "
#define N 1000
struct T {
    int x;
    int y;
};

extern \"C\" __global__ void add_vectors(T *a, T *b, T *c)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < N) {
        c[id].x = a[id].x + b[id].x;
        c[id].y = a[id].y + b[id].y;
    }
}
";

#[derive(Debug)]
#[warn(dead_code)]
struct T {
    x: i32,
    y: i32,
}

unsafe impl DeviceRepr for T {}

fn main() -> Result<(), DriverError> {
    let start = std::time::Instant::now();
    let ptx = compile_ptx(SRC).unwrap();
    println!("Compilation succeeded in {:?}", start.elapsed());

    let dev = CudaDevice::new(0)?;
    println!("Built in {:?}", start.elapsed());

    dev.load_ptx(ptx, "add_vectors", &["add_vectors"])?;
    let f = dev.get_func("add_vectors", "add_vectors").unwrap();
    println!("Loaded in {:?}", start.elapsed());

    let ah = [T { x: 1, y: 2 }, T { x: 2, y: 3 }, T { x: 3, y: 4 }];
    let bh = [T { x: 1, y: 2 }, T { x: 2, y: 3 }, T { x: 3, y: 4 }];
    let mut ch = [T { x: 0, y: 0 }, T { x: 0, y: 0 }, T { x: 0, y: 0 }];
    let a_dev = dev.htod_sync_copy(&ah)?;
    let b_dev = dev.htod_sync_copy(&bh)?;
    let mut c_dev = dev.htod_sync_copy(&ch)?;

    println!("Copied in {:?}", start.elapsed());
    let cfg = LaunchConfig {
        block_dim: (3, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let _res = unsafe { f.launch(cfg, (&a_dev, &b_dev, &mut c_dev)) };
    dev.dtoh_sync_copy_into(&c_dev, &mut ch)?;
    println!("Adding vectors {:?} and {:?}.", ah, bh);
    println!("Found {:?} in {:?}", ch, start.elapsed());
    Ok(())
}
