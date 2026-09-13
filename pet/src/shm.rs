//! Memfd-backed double-buffered shared memory pool for wl_shm.
//!
//! Holds two contiguous ARGB8888 buffer regions in one mmap. We tell the
//! compositor about both as `wl_buffer`s and ping-pong between them every
//! frame. The compositor releases the previous buffer asynchronously; if we
//! ever try to write to one before it's released we'd tear, but at our frame
//! rate (~30 fps) double-buffering is sufficient.

use std::ffi::CString;
use std::os::fd::{AsFd, BorrowedFd, FromRawFd, OwnedFd};
use std::ptr::NonNull;

use anyhow::{anyhow, Context, Result};
use wayland_client::protocol::wl_buffer::WlBuffer;
use wayland_client::protocol::wl_shm::{Format, WlShm};
use wayland_client::protocol::wl_shm_pool::WlShmPool;
use wayland_client::{Dispatch, QueueHandle};

pub struct ShmPool {
    // Held so the memfd lives as long as the mapping it backs.
    _fd: OwnedFd,
    ptr: NonNull<u8>,
    map_len: usize,
    pub width: u32,
    pub height: u32,
    pub stride: usize,
    pub region_size: usize,
    pub pool: WlShmPool,
    pub buffers: [WlBuffer; 2],
}

// Safety: ptr is valid for the lifetime of the mapping; we use it only via
// mutable slices in render paths gated by buf_index.
unsafe impl Send for ShmPool {}

impl ShmPool {
    pub fn create<D>(
        shm: &WlShm,
        qh: &QueueHandle<D>,
        width: u32,
        height: u32,
    ) -> Result<Self>
    where
        D: Dispatch<WlShmPool, ()> + Dispatch<WlBuffer, ()> + 'static,
    {
        let stride = width as usize * 4;
        let region_size = stride * height as usize;
        let map_len = region_size * 2;

        let fd = memfd_create("lanty-pet-shm")?;
        unsafe {
            if libc::ftruncate(fd.as_fd().as_raw_fd_int(), map_len as libc::off_t) < 0 {
                return Err(anyhow!(
                    "ftruncate: {}",
                    std::io::Error::last_os_error()
                ));
            }
        }

        let raw = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                map_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_fd().as_raw_fd_int(),
                0,
            )
        };
        if raw == libc::MAP_FAILED {
            return Err(anyhow!("mmap: {}", std::io::Error::last_os_error()));
        }
        let ptr = NonNull::new(raw.cast::<u8>())
            .ok_or_else(|| anyhow!("mmap returned null"))?;

        let pool = shm.create_pool(fd.as_fd(), map_len as i32, qh, ());
        let buf0 = pool.create_buffer(
            0,
            width as i32,
            height as i32,
            stride as i32,
            Format::Argb8888,
            qh,
            (),
        );
        let buf1 = pool.create_buffer(
            region_size as i32,
            width as i32,
            height as i32,
            stride as i32,
            Format::Argb8888,
            qh,
            (),
        );

        Ok(Self {
            _fd: fd,
            ptr,
            map_len,
            width,
            height,
            stride,
            region_size,
            pool,
            buffers: [buf0, buf1],
        })
    }

    /// Mutable slice for one of the two buffer regions. The caller is
    /// responsible for not writing to a region that the compositor still
    /// holds (we double-buffer to avoid this in practice).
    pub fn region_mut(&mut self, index: usize) -> &mut [u8] {
        let offset = index * self.region_size;
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr.as_ptr().add(offset), self.region_size)
        }
    }
}

impl Drop for ShmPool {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr.as_ptr().cast(), self.map_len);
        }
        for b in &self.buffers {
            b.destroy();
        }
        self.pool.destroy();
    }
}

fn memfd_create(name: &str) -> Result<OwnedFd> {
    let cname = CString::new(name).context("memfd name")?;
    let fd = unsafe {
        libc::syscall(
            libc::SYS_memfd_create,
            cname.as_ptr(),
            libc::MFD_CLOEXEC,
        ) as libc::c_int
    };
    if fd < 0 {
        return Err(anyhow!(
            "memfd_create: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

trait AsRawFdInt {
    fn as_raw_fd_int(&self) -> libc::c_int;
}

impl AsRawFdInt for BorrowedFd<'_> {
    fn as_raw_fd_int(&self) -> libc::c_int {
        use std::os::fd::AsRawFd;
        self.as_raw_fd()
    }
}
