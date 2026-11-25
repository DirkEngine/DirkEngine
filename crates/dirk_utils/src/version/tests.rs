use super::*;

// -- Construction & accessors --

#[test]
fn test_fields_roundtrip() {
    let v = Version::new(1, 2, 3);
    assert_eq!(v.major(), 1);
    assert_eq!(v.minor(), 2);
    assert_eq!(v.patch(), 3);
}

#[test]
fn test_zero_constant() {
    assert_eq!(Version::ZERO.major(), 0);
    assert_eq!(Version::ZERO.minor(), 0);
    assert_eq!(Version::ZERO.patch(), 0);
}

/// major/minor: 10 bits → max 1023 ; patch: 12 bits → max 4095
#[test]
fn test_max_field_values() {
    let v = Version::new(1023, 1023, 4095);
    assert_eq!(v.major(), 1023);
    assert_eq!(v.minor(), 1023);
    assert_eq!(v.patch(), 4095);
}

#[test]
#[should_panic(expected = "major version must fit in 10 bits")]
fn test_major_overflow_panics() {
    let _ = Version::new(1024, 0, 0);
}

#[test]
#[should_panic(expected = "patch version must fit in 12 bits")]
fn test_patch_overflow_panics() {
    let _ = Version::new(0, 0, 4096);
}

// -- Bumping --

#[test]
fn test_bump_major_resets_minor_and_patch() {
    assert_eq!(
        Version::new(1, 5, 3).bump_major(),
        Some(Version::new(2, 0, 0))
    );
}

#[test]
fn test_bump_minor_resets_patch() {
    assert_eq!(
        Version::new(1, 5, 3).bump_minor(),
        Some(Version::new(1, 6, 0))
    );
}

#[test]
fn test_bump_patch() {
    assert_eq!(
        Version::new(1, 5, 3).bump_patch(),
        Some(Version::new(1, 5, 4))
    );
}

#[test]
fn test_bump_at_limit_returns_none() {
    assert_eq!(Version::new(1023, 5, 3).bump_major(), None);
    assert_eq!(Version::new(1, 1023, 3).bump_minor(), None);
    assert_eq!(Version::new(1, 5, 4095).bump_patch(), None);
    assert_eq!(
        Version::new(1, 1023, 4095).bump_major(),
        Some(Version::new(2, 0, 0))
    );
}

// -- Ordering --

#[test]
fn test_ordering() {
    assert!(Version::new(2, 0, 0) > Version::new(1, 9, 9));
    assert!(Version::new(1, 1, 0) > Version::new(1, 0, 99));
    assert!(Version::new(1, 0, 1) > Version::new(1, 0, 0));
    assert_eq!(Version::new(1, 2, 3), Version::new(1, 2, 3));
}

// -- Compatibility --

#[test]
fn test_compatible_same_major_higher_minor() {
    assert!(Version::new(1, 3, 0).is_compatible_with(Version::new(1, 2, 0)));
}

#[test]
fn test_incompatible_different_major() {
    assert!(!Version::new(2, 0, 0).is_compatible_with(Version::new(1, 0, 0)));
}

#[test]
fn test_incompatible_older_version() {
    assert!(!Version::new(1, 1, 0).is_compatible_with(Version::new(1, 2, 0)));
}

#[test]
fn test_prerelease() {
    assert!(Version::new(0, 9, 0).is_prerelease());
    assert!(!Version::new(1, 0, 0).is_prerelease());
}

// -- Formatting & parsing --

#[test]
fn test_display() {
    assert_eq!(Version::new(1, 2, 3).to_string(), "1.2.3");
}

#[test]
fn test_parse_plain() {
    let v: Version = "1.2.3".parse().expect("valid version should parse");
    assert_eq!((v.major(), v.minor(), v.patch()), (1, 2, 3));
}

#[test]
fn test_parse_with_v_prefix() {
    let v: Version = "v2.0.0".parse().expect("valid version should parse");
    assert_eq!(v.major(), 2);
}

#[test]
fn test_parse_invalid() {
    for input in ["1024.0.0", "0.1024.0", "0.0.4096", "4294967295.0.0"] {
        assert!(input.parse::<Version>().is_err(), "{input}");
    }
    assert!("1.2".parse::<Version>().is_err());
    assert!("abc".parse::<Version>().is_err());
    assert!("1.x.3".parse::<Version>().is_err());
}

#[test]
fn test_parse_display_roundtrip() {
    let original = Version::new(3, 14, 42);
    let roundtripped: Version = original
        .to_string()
        .parse()
        .expect("displayed version should parse");
    assert_eq!(original, roundtripped);
}

// -- Raw conversion --

#[test]
fn test_from_into_u32() {
    let v = Version::new(1, 2, 3);
    let raw: u32 = v.into();
    let back = Version::from(raw);
    assert_eq!(v, back);
}
