//! Internal macros shared by the query and system modules.

/// Invokes `$m` once for every tuple arity from 1 to 12. Each element is a
/// type identifier paired with a binding identifier, e.g. `$m!(A a, B b)`.
macro_rules! all_tuples {
    ($m:ident) => {
        $m!(A a);
        $m!(A a, B b);
        $m!(A a, B b, C c);
        $m!(A a, B b, C c, D d);
        $m!(A a, B b, C c, D d, E e);
        $m!(A a, B b, C c, D d, E e, F f);
        $m!(A a, B b, C c, D d, E e, F f, G g);
        $m!(A a, B b, C c, D d, E e, F f, G g, H h);
        $m!(A a, B b, C c, D d, E e, F f, G g, H h, I i);
        $m!(A a, B b, C c, D d, E e, F f, G g, H h, I i, J j);
        $m!(A a, B b, C c, D d, E e, F f, G g, H h, I i, J j, K k);
        $m!(A a, B b, C c, D d, E e, F f, G g, H h, I i, J j, K k, L l);
    };
}

/// Seals the public query and system traits: they are implemented only by
/// this crate, so the storage behind them can change freely. Each trait has
/// its own seal, so sealing a type for one trait never lets other crates
/// implement another trait for it.
pub(crate) mod sealed {
    pub trait QueryData {}
    pub trait ReadOnlyQueryData {}
    pub trait QueryFilter {}
    pub trait SystemParam {}
    pub trait Mutability {}
    pub trait DerivedArg {}

    macro_rules! impl_sealed_for_tuple {
        ($($ty:ident $binding:ident),*) => {
            impl<$($ty),*> QueryData for ($($ty,)*) {}
            impl<$($ty),*> ReadOnlyQueryData for ($($ty,)*) {}
            impl<$($ty),*> QueryFilter for ($($ty,)*) {}
            impl<$($ty),*> SystemParam for ($($ty,)*) {}
        };
    }
    impl_sealed_for_tuple!();
    all_tuples!(impl_sealed_for_tuple);
}
