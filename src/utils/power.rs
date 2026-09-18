// SPDX-License-Identifier: GPL-3.0-only

//! Power-aware animation frame pacing.
//!
//! Compositor-driven animations (workspace transitions, dock hover, overview,
//! window open/close) normally re-render at the full output refresh rate.
//! On battery this spends significant power for little visual benefit on
//! high-refresh displays, so animation-driven redraws can be capped to a
//! lower frame rate. Client content and cursor updates are never capped.
//!
//! The cap is controlled by (in order of precedence):
//! - `COSMIC_ANIMATION_FPS`: `N` (> 0) caps animations to N fps everywhere,
//!   `0` disables capping entirely.
//! - Otherwise logind's `OnBattery` property: on battery animations are
//!   capped to [`BATTERY_CAP_FPS`], on AC they run at full refresh rate.

use std::sync::{
    OnceLock,
    atomic::{AtomicU64, Ordering},
};

use tracing::{debug, warn};

/// Animation frame cap applied while on battery power.
const BATTERY_CAP_FPS: u64 = 60;

/// Minimum interval between animation-driven frames, in milliseconds.
/// 0 = no capping (animations render at the output refresh rate).
static ANIMATION_FRAME_INTERVAL_MS: AtomicU64 = AtomicU64::new(0);

/// Returns the minimum interval between animation-driven frames, if capping
/// is active.
pub fn animation_frame_interval() -> Option<std::time::Duration> {
    match ANIMATION_FRAME_INTERVAL_MS.load(Ordering::Relaxed) {
        0 => None,
        ms => Some(std::time::Duration::from_millis(ms)),
    }
}

fn set_interval_from_fps(fps: u64) {
    let ms = if fps == 0 { 0 } else { 1000 / fps };
    ANIMATION_FRAME_INTERVAL_MS.store(ms, Ordering::Relaxed);
}

#[derive(Debug, Clone, Copy)]
enum CapMode {
    /// Always cap to the given fps (env override).
    Fixed(u64),
    /// Follow the battery status.
    Battery,
    /// Never cap (env override).
    Off,
}

fn cap_mode() -> CapMode {
    static MODE: OnceLock<CapMode> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("COSMIC_ANIMATION_FPS") {
        Ok(value) => match value.trim().parse::<u64>() {
            Ok(0) => CapMode::Off,
            Ok(fps) => CapMode::Fixed(fps),
            Err(_) => {
                warn!("Ignoring invalid COSMIC_ANIMATION_FPS={value:?}, expected a number");
                CapMode::Battery
            }
        },
        Err(_) => CapMode::Battery,
    })
}

/// Spawn the background thread tracking the power source (if needed for the
/// configured cap mode). Cheap no-op when capping is fixed or disabled.
pub fn spawn_power_watcher() {
    match cap_mode() {
        CapMode::Fixed(fps) => {
            set_interval_from_fps(fps);
            debug!(fps, "Animation frame cap set via COSMIC_ANIMATION_FPS");
        }
        CapMode::Off => {
            debug!("Animation frame capping disabled via COSMIC_ANIMATION_FPS");
        }
        CapMode::Battery => {
            std::thread::Builder::new()
                .name("cosmic-power-watcher".into())
                .spawn(watch_battery)
                .expect("Failed to spawn power watcher thread");
        }
    }
}

fn watch_battery() {
    use futures_util::StreamExt;
    use zbus::fdo::PropertiesProxy;

    let apply = |on_battery: bool| {
        let fps = if on_battery { BATTERY_CAP_FPS } else { 0 };
        set_interval_from_fps(fps);
        debug!(on_battery, "Animation frame cap updated");
    };

    let conn = match futures_executor::block_on(zbus::Connection::system()) {
        Ok(conn) => conn,
        Err(err) => {
            warn!(
                ?err,
                "No system dbus connection, cannot watch battery status"
            );
            return;
        }
    };

    let proxy = match futures_executor::block_on(PropertiesProxy::new(
        &conn,
        "org.freedesktop.login1",
        "/org/freedesktop/login1",
    )) {
        Ok(proxy) => proxy,
        Err(err) => {
            warn!(
                ?err,
                "Failed to create logind proxy, animations stay uncapped"
            );
            return;
        }
    };

    let manager = match futures_executor::block_on(zbus::Proxy::new(
        &conn,
        "org.freedesktop.login1",
        "/org/freedesktop/login1",
        "org.freedesktop.login1.Manager",
    )) {
        Ok(proxy) => proxy,
        Err(err) => {
            warn!(
                ?err,
                "Failed to create logind proxy, animations stay uncapped"
            );
            return;
        }
    };

    let read_on_battery = || async {
        manager
            .get_property::<bool>("OnBattery")
            .await
            .map_err(|err| {
                tracing::debug!(?err, "Failed to read OnBattery");
                err
            })
    };

    match futures_executor::block_on(read_on_battery()) {
        Ok(on_battery) => apply(on_battery),
        Err(_) => {
            // e.g. desktops without batteries: keep running uncapped
            return;
        }
    }

    let stream = match futures_executor::block_on(proxy.receive_properties_changed()) {
        Ok(stream) => stream,
        Err(err) => {
            warn!(?err, "Failed to watch OnBattery, animations stay uncapped");
            return;
        }
    };

    let mut stream = stream;
    while let Some(change) = futures_executor::block_on(stream.next()) {
        let args = match change.args() {
            Ok(args) => args,
            Err(err) => {
                warn!(?err, "Malformed PropertiesChanged from logind");
                continue;
            }
        };
        if args.interface_name() != "org.freedesktop.login1.Manager" {
            continue;
        }
        if args.changed_properties().contains_key("OnBattery")
            || args.invalidated_properties().contains(&"OnBattery")
        {
            match futures_executor::block_on(read_on_battery()) {
                Ok(on_battery) => apply(on_battery),
                Err(err) => warn!(?err, "Failed to read OnBattery"),
            }
        }
    }
}
