// SPDX-License-Identifier: GPL-3.0-only

use crate::state::State;
use smithay::wayland::shell::xdg::{
    ToplevelSurface,
    dialog::{ToplevelDialogHint, XdgDialogHandler},
};

impl XdgDialogHandler for State {
    fn dialog_hint_changed(&mut self, toplevel: ToplevelSurface, hint: ToplevelDialogHint) {
        let dialog = toplevel.wl_surface().clone();
        if hint == ToplevelDialogHint::Modal {
            self.common.modal_dialogs.set(dialog, toplevel.parent());
        } else {
            self.common.modal_dialogs.remove(&dialog);
        }
    }
}
