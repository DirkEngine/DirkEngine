macro_rules! define_flags {
    (
        $(#[$meta:meta])*
        pub struct $name:ident($storage:ty) {
            $($(#[$flag_meta:meta])* const $flag:ident = $value:expr;)+
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
        pub struct $name($storage);

        impl $name {
            /// No flags.
            pub const NONE: Self = Self(0);
            $($(#[$flag_meta])* pub const $flag: Self = Self($value);)+
            /// All flags declared by this type.
            pub const ALL: Self = Self(0 $(| $value)+);

            /// Creates a flag set if `bits` contains only declared flags.
            #[must_use]
            pub const fn from_bits(bits: $storage) -> Option<Self> {
                if bits & !Self::ALL.0 == 0 {
                    Some(Self(bits))
                } else {
                    None
                }
            }

            /// Creates a flag set after discarding undeclared bits.
            #[must_use]
            pub const fn from_bits_truncate(bits: $storage) -> Self {
                Self(bits & Self::ALL.0)
            }

            /// Returns true when all flags in `other` are present.
            #[must_use]
            pub const fn contains(self, other: Self) -> bool {
                self.0 & other.0 == other.0
            }

            /// Returns true when any flag in `other` is present.
            #[must_use]
            pub const fn intersects(self, other: Self) -> bool {
                self.0 & other.0 != 0
            }

            /// Returns true when no flags are set.
            #[must_use]
            pub const fn is_empty(self) -> bool {
                self.0 == 0
            }

            /// Returns true when every declared flag is set.
            #[must_use]
            pub const fn is_all(self) -> bool {
                self.contains(Self::ALL)
            }

            /// Returns the underlying bit representation.
            #[must_use]
            pub const fn bits(self) -> $storage {
                self.0
            }

            /// Returns the union of `self` and `other` in a const context.
            #[must_use]
            pub const fn union(self, other: Self) -> Self {
                Self(self.0 | other.0)
            }

            /// Returns the intersection of `self` and `other` in a const context.
            #[must_use]
            pub const fn intersection(self, other: Self) -> Self {
                Self(self.0 & other.0)
            }

            /// Returns the flags in `self` that are not in `other`.
            #[must_use]
            pub const fn difference(self, other: Self) -> Self {
                Self(self.0 & !other.0)
            }

            /// Returns the flags set in exactly one of the two flag sets.
            #[must_use]
            pub const fn symmetric_difference(self, other: Self) -> Self {
                Self(self.0 ^ other.0)
            }

            /// Adds all flags in `other`.
            pub fn insert(&mut self, other: Self) {
                self.0 |= other.0;
            }

            /// Removes all flags in `other`.
            pub fn remove(&mut self, other: Self) {
                self.0 &= !other.0;
            }

            /// Toggles all flags in `other`.
            pub fn toggle(&mut self, other: Self) {
                self.0 ^= other.0;
            }
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str(stringify!($name))?;
                formatter.write_str("(")?;
                let mut first = true;
                $(
                    if self.contains(Self::$flag) {
                        if !first {
                            formatter.write_str(" | ")?;
                        }
                        formatter.write_str(stringify!($flag))?;
                        first = false;
                    }
                )+
                let unknown = self.0 & !Self::ALL.0;
                if unknown != 0 {
                    if !first {
                        formatter.write_str(" | ")?;
                    }
                    write!(formatter, "{unknown:#x}")?;
                }
                formatter.write_str(")")
            }
        }

        impl std::ops::BitOr for $name {
            type Output = Self;

            fn bitor(self, rhs: Self) -> Self::Output {
                Self(self.0 | rhs.0)
            }
        }

        impl std::ops::BitOrAssign for $name {
            fn bitor_assign(&mut self, rhs: Self) {
                self.0 |= rhs.0;
            }
        }

        impl std::ops::BitAnd for $name {
            type Output = Self;

            fn bitand(self, rhs: Self) -> Self::Output {
                Self(self.0 & rhs.0)
            }
        }

        impl std::ops::BitAndAssign for $name {
            fn bitand_assign(&mut self, rhs: Self) {
                self.0 &= rhs.0;
            }
        }

        impl std::ops::BitXor for $name {
            type Output = Self;

            fn bitxor(self, rhs: Self) -> Self::Output {
                Self(self.0 ^ rhs.0)
            }
        }

        impl std::ops::BitXorAssign for $name {
            fn bitxor_assign(&mut self, rhs: Self) {
                self.0 ^= rhs.0;
            }
        }

        impl std::ops::Sub for $name {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self::Output {
                self.difference(rhs)
            }
        }

        impl std::ops::SubAssign for $name {
            fn sub_assign(&mut self, rhs: Self) {
                self.remove(rhs);
            }
        }

        impl std::ops::Not for $name {
            type Output = Self;

            fn not(self) -> Self::Output {
                Self(!self.0 & Self::ALL.0)
            }
        }
    };
}

pub(crate) use define_flags;
