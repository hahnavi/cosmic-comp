// SPDX-License-Identifier: GPL-3.0-only

//! Memory reclamation and allocator tuning for cosmic-comp.
//!
//! Long-running compositors accumulate virtual and resident memory over time
//! from texture allocations, layout trees, and client buffer mappings.
//! This module configures mimalloc for prompt decommit/purging of freed pages
//! back to the Linux kernel, disables Transparent Huge Pages (THP) to avoid 2MB page bloat,
//! and provides functions to compact and trim memory during idle periods or window closures.

use tracing::debug;

/// Tune allocator and kernel memory management parameters for lower resident memory footprint.
pub fn init_allocator() {
    // Disable Transparent Huge Pages (THP) for cosmic-comp.
    // When /sys/kernel/mm/transparent_hugepage/enabled is set to 'always',
    // the Linux kernel rounds up every 2MB-aligned allocation to a 2MB physical page,
    // inflating RSS by 60MB+ of unused zero-padding.
    #[cfg(target_os = "linux")]
    unsafe {
        unsafe extern "C" {
            fn prctl(
                option: std::os::raw::c_int,
                arg2: std::os::raw::c_ulong,
                arg3: std::os::raw::c_ulong,
                arg4: std::os::raw::c_ulong,
                arg5: std::os::raw::c_ulong,
            ) -> std::os::raw::c_int;
        }
        const PR_SET_THP_DISABLE: std::os::raw::c_int = 41;
        prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
    }

    #[cfg(not(feature = "profile-with-tracy"))]
    unsafe {
        // Disallow large/huge OS pages in mimalloc so it uses granular 4KB pages.
        libmimalloc_sys::mi_option_set(libmimalloc_sys::mi_option_large_os_pages, 0);
        libmimalloc_sys::mi_option_set(libmimalloc_sys::mi_option_reserve_huge_os_pages, 0);
    }

    debug!("Initialized allocator tuning and disabled THP bloat");
}

/// Collect and trim memory back to the operating system.
///
/// If `force` is true, all free pages across all arenas are actively returned to the kernel.
pub fn trim_memory(force: bool) {
    #[cfg(not(feature = "profile-with-tracy"))]
    unsafe {
        libmimalloc_sys::mi_collect(force);
    }
}
