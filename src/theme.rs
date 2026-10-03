// insert into the event loop, a watcher for the theme & theme mode for changes

// update a Arc<Mutex<Theme>> in the state on change of the theme and mark all interfaces for a redraw.

use calloop::LoopHandle;
use cosmic::cosmic_theme::{Theme, ThemeMode, palette};

use crate::state::State;

pub(crate) fn _group_color(theme: &Theme) -> [f32; 3] {
    let neutral_8 = theme.palette.neutral_8;
    [neutral_8.red, neutral_8.green, neutral_8.blue]
}

pub(crate) fn active_window_hint(theme: &Theme) -> palette::Srgba {
    if let Some(hint) = theme.window_hint {
        palette::Srgba::from(hint)
    } else {
        theme.accent_color()
    }
}

pub fn watch_theme(handle: LoopHandle<'_, State>) -> Result<(), cosmic_config::Error> {
    let (ping_tx, ping_rx) = calloop::ping::make_ping().unwrap();
    let _ = ThemeMode::config()?;
    let _ = Theme::dark_config()?;
    let _ = Theme::light_config()?;

    if let Err(e) = handle.insert_source(ping_rx, move |_, _, state| {
        let new_theme = cosmic::theme::system_preference();
        let theme = &mut state.common.theme;

        if theme.theme_type != new_theme.theme_type {
            *theme = new_theme;
            let mut workspace_guard = state.common.workspace_state.update();
            state.common.shell.write().set_theme(
                theme.clone(),
                &state.common.xdg_activation_state,
                &mut workspace_guard,
            );
        }
    }) {
        tracing::error!("{e}");
    };

    let cosmic_dir = xdg::BaseDirectories::with_prefix("cosmic")
        .get_config_home()
        .unwrap_or_else(|| std::path::PathBuf::from("/etc/cosmic"));

    let ping_tx_clone = ping_tx.clone();
    let cosmic_dir_clone = cosmic_dir.clone();
    let mut watcher =
        match notify::recommended_watcher(move |event_res: Result<notify::Event, notify::Error>| {
            if let Ok(event) = event_res {
                match &event.kind {
                    notify::EventKind::Access(_)
                    | notify::EventKind::Modify(notify::event::ModifyKind::Metadata(_)) => {}
                    _ => {
                        for path in &event.paths {
                            if let Ok(rel) = path.strip_prefix(&cosmic_dir_clone) {
                                if let Some(std::path::Component::Normal(app)) =
                                    rel.components().next()
                                {
                                    if app.to_str().map_or(false, |s| {
                                        s.starts_with("com.system76.CosmicTheme")
                                    }) {
                                        ping_tx_clone.ping();
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }) {
            Ok(w) => w,
            Err(err) => {
                tracing::warn!(?err, "Failed to create unified theme watcher");
                return Ok(());
            }
        };

    use notify::Watcher;
    let _ = std::fs::create_dir_all(&cosmic_dir);
    let _ = watcher.watch(&cosmic_dir, notify::RecursiveMode::Recursive);

    std::mem::forget(watcher);

    Ok(())
}
