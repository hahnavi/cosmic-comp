// SPDX-License-Identifier: GPL-3.0-only

//! Tracks modal dialogs and their parent toplevels.

use indexmap::IndexMap;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;

/// Modal dialog relationships requested by clients.
/// A parent stays unfocused while its modal child is open.
#[derive(Debug, Default)]
pub struct ModalDialogs {
    /// dialog surface -> parent surface
    dialogs: IndexMap<WlSurface, WlSurface>,
}

impl ModalDialogs {
    /// Register `dialog` as a modal child of `parent`.
    /// Clears any prior relation when the dialog is no longer modal.
    pub fn set(&mut self, dialog: WlSurface, parent: Option<WlSurface>) {
        match parent {
            Some(parent) => {
                self.dialogs.insert(dialog, parent);
            }
            None => {
                self.dialogs.shift_remove(&dialog);
            }
        }
    }

    /// Remove the modal state of `dialog`.
    pub fn remove(&mut self, dialog: &WlSurface) {
        self.dialogs.shift_remove(dialog);
    }

    /// Remove all dialogs attached to `parent`.
    pub fn remove_parent(&mut self, parent: &WlSurface) {
        self.dialogs
            .retain(|_, dialog_parent| dialog_parent != parent);
    }

    /// Return the most recent modal dialog for `parent`, if any.
    pub fn dialog_for(&self, parent: &WlSurface) -> Option<&WlSurface> {
        self.dialogs
            .iter()
            .rev()
            .find_map(|(dialog, dialog_parent)| (dialog_parent == parent).then_some(dialog))
    }
}
