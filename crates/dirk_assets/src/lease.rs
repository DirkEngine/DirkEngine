//! Keeping assets loaded for as long as something needs them.

use std::sync::{Arc, Weak};

use parking_lot::Mutex;
use tracing::error;

use crate::{Asset, AssetHandle, AssetLoad, Handle};

/// Keeps an asset loaded for as long as the lease lives.
///
/// Created by [`AssetRegistry::lease`](crate::AssetRegistry::lease). The load
/// runs on the worker pool and completes during
/// [`AssetRegistry::tick`](crate::AssetRegistry::tick), which also logs
/// failures, so holders never poll. Dropping the lease releases the asset.
pub struct AssetLease<T: Asset> {
    asset: AssetHandle,
    state: Arc<Mutex<LeaseState<T>>>,
}

enum LeaseState<T: Asset> {
    Loading(AssetLoad<T>),
    Loaded(Handle<T>),
    Failed,
}

impl<T: Asset> AssetLease<T> {
    /// Starts leasing `asset`, returning the lease and the registry's weak
    /// reference used to complete the load.
    pub(crate) fn new(asset: AssetHandle, load: AssetLoad<T>) -> (Self, Box<dyn PendingLease>) {
        let state = Arc::new(Mutex::new(LeaseState::Loading(load)));
        let pending = Box::new(Pending {
            asset: asset.clone(),
            state: Arc::downgrade(&state),
        });
        (Self { asset, state }, pending)
    }

    /// Returns the leased asset's handle.
    #[must_use]
    pub fn asset(&self) -> &AssetHandle {
        &self.asset
    }

    /// Returns the loaded asset, or `None` while loading or after a failure.
    #[must_use]
    pub fn get(&self) -> Option<Handle<T>> {
        match &*self.state.lock() {
            LeaseState::Loaded(handle) => Some(handle.clone()),
            LeaseState::Loading(_) | LeaseState::Failed => None,
        }
    }
}

/// A lease the registry completes during its tick.
pub(crate) trait PendingLease: Send + Sync {
    /// Advances the load, returning `false` once nothing is left to do.
    fn poll(&self) -> bool;
}

struct Pending<T: Asset> {
    asset: AssetHandle,
    state: Weak<Mutex<LeaseState<T>>>,
}

impl<T: Asset> PendingLease for Pending<T> {
    fn poll(&self) -> bool {
        // A dropped lease has nothing left to load.
        let Some(state) = self.state.upgrade() else {
            return false;
        };
        let mut state = state.lock();
        let LeaseState::Loading(load) = &mut *state else {
            return false;
        };
        match load.try_poll() {
            None => true,
            Some(Ok(handle)) => {
                *state = LeaseState::Loaded(handle);
                false
            }
            Some(Err(error)) => {
                error!(asset = %self.asset, ?error, "failed to load leased asset");
                *state = LeaseState::Failed;
                false
            }
        }
    }
}
