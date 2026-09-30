//! Tests for ordering systems from the data they access.

use std::{any::type_name, cell::RefCell};

use dirk_universe::{
    UniverseBuilder,
    prelude::*,
    schedule::{Dependency, Reason, ScheduleError},
};

#[derive(Debug, Component)]
struct A(i32);
#[derive(Debug, Component)]
struct B;
#[derive(Debug, Component)]
struct C;

thread_local! {
    static LOG: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
}

/// Records that the calling system ran.
fn log(name: &'static str) {
    LOG.with_borrow_mut(|log| log.push(name));
}

/// Returns and clears the systems that ran, in order.
fn take_log() -> Vec<&'static str> {
    LOG.with_borrow_mut(std::mem::take)
}

fn write_a(mut query: Query<&mut A>) {
    log("write_a");
    for mut a in &mut query {
        a.0 += 1;
    }
}

fn read_a_write_b(_query: Query<(&A, &mut B)>) {
    log("read_a_write_b");
}

fn read_b(_query: Query<&B>) {
    log("read_b");
}

fn read_c(_query: Query<&C>) {
    log("read_c");
}

fn universe_with_entity(builder: UniverseBuilder) -> UniverseBuilder {
    builder.with_world(
        World::builder("w").with_entity(
            Entity::builder()
                .with_component(A(0))
                .with_component(B)
                .with_component(C),
        ),
    )
}

/// Returns every ordering of `0..len`.
fn permutations(len: usize) -> Vec<Vec<usize>> {
    if len == 0 {
        return vec![Vec::new()];
    }
    let mut result = Vec::new();
    for rest in permutations(len - 1) {
        for position in 0..=rest.len() {
            let mut permutation = rest.clone();
            permutation.insert(position, len - 1);
            result.push(permutation);
        }
    }
    result
}

type Register = fn(UniverseBuilder) -> UniverseBuilder;

/// Builds a universe registering `systems` in the order given by `order`.
fn build(systems: &[Register], order: &[usize]) -> Result<Universe, ScheduleError> {
    order
        .iter()
        .fold(
            universe_with_entity(Universe::builder()),
            |builder, &index| systems[index](builder),
        )
        .build()
}

fn position(log: &[&str], name: &str) -> usize {
    log.iter()
        .position(|entry| *entry == name)
        .unwrap_or_else(|| panic!("{name} did not run"))
}

#[test]
fn readers_run_after_writers_whatever_the_registration_order() {
    let systems: [Register; 4] = [
        |b| b.with_system(read_b),
        |b| b.with_system(read_a_write_b),
        |b| b.with_system(read_c),
        |b| b.with_system(write_a),
    ];
    for order in permutations(systems.len()) {
        let mut universe = build(&systems, &order).expect("systems should schedule");
        take_log();
        universe.tick(0.0);
        let log = take_log();

        assert_eq!(log.len(), 4, "every system runs once in {order:?}");
        assert!(position(&log, "write_a") < position(&log, "read_a_write_b"));
        assert!(position(&log, "read_a_write_b") < position(&log, "read_b"));
    }
}

#[test]
fn unrelated_systems_keep_their_registration_order() {
    let mut universe = universe_with_entity(Universe::builder())
        .with_system(read_c)
        .with_system(read_b)
        .build()
        .expect("systems should schedule");
    take_log();
    universe.tick(0.0);
    assert_eq!(take_log(), vec!["read_c", "read_b"]);
}

#[test]
fn a_reader_registered_first_sees_writes_from_the_same_tick() {
    thread_local! {
        static SEEN: RefCell<Vec<i32>> = const { RefCell::new(Vec::new()) };
    }
    fn record(query: Query<&A>) {
        SEEN.with_borrow_mut(|seen| seen.extend(query.iter().map(|a| a.0)));
    }
    let mut universe = universe_with_entity(Universe::builder())
        .with_system(record)
        .with_system(write_a)
        .build()
        .expect("systems should schedule");
    for _ in 0..3 {
        universe.tick(0.0);
    }
    assert_eq!(SEEN.with_borrow(Clone::clone), vec![1, 2, 3]);
}

#[test]
fn filters_and_deltas_order_their_systems_after_writers() {
    fn changed(_query: Query<Entity, Changed<A>>) {
        log("changed");
    }
    fn delta(_query: Query<(Entity, Delta<A>)>) {
        log("delta");
    }
    let mut universe = universe_with_entity(Universe::builder())
        .with_system(changed)
        .with_system(delta)
        .with_system(write_a)
        .build()
        .expect("systems should schedule");
    take_log();
    universe.tick(0.0);
    assert_eq!(take_log(), vec!["write_a", "changed", "delta"]);
}

#[test]
fn lagged_readers_run_before_writers_and_see_last_tick() {
    thread_local! {
        static SEEN: RefCell<Vec<i32>> = const { RefCell::new(Vec::new()) };
    }
    fn record(query: Query<Lagged<&A>>) {
        SEEN.with_borrow_mut(|seen| seen.extend(query.iter().map(|a| a.0)));
    }
    let mut universe = universe_with_entity(Universe::builder())
        .with_system(write_a)
        .with_system(record)
        .build()
        .expect("systems should schedule");
    for _ in 0..3 {
        universe.tick(0.0);
    }
    assert_eq!(SEEN.with_borrow(Clone::clone), vec![0, 1, 2]);
    assert_eq!(
        universe.schedule().systems()[1].after,
        vec![Dependency {
            system: type_name_of(record),
            reason: Reason::ReadsLagged(type_name::<A>()),
        }]
    );
}

fn write_a_read_b(_query: Query<(&mut A, &B)>) {}

fn write_b_read_a(_query: Query<(&mut B, &A)>) {}

fn write_b_read_lagged_a(_query: Query<(&mut B, Lagged<&A>)>) {}

fn type_name_of<T>(_: T) -> &'static str {
    type_name::<T>()
}

#[test]
fn mutual_dependencies_are_a_cycle_naming_each_system_and_component() {
    let error = Universe::builder()
        .with_system(write_a_read_b)
        .with_system(write_b_read_a)
        .build()
        .err()
        .expect("mutual dependencies should not schedule");

    assert_eq!(
        error,
        ScheduleError::Cycle(vec![
            Dependency {
                system: type_name_of(write_a_read_b),
                reason: Reason::Writes(type_name::<A>()),
            },
            Dependency {
                system: type_name_of(write_b_read_a),
                reason: Reason::Writes(type_name::<B>()),
            },
        ])
    );
    let message = error.to_string();
    assert!(message.contains(&format!(
        "`{}` runs after `{}`, which writes `{}`",
        type_name_of(write_b_read_a),
        type_name_of(write_a_read_b),
        type_name::<A>()
    )));
}

#[test]
fn a_lagged_read_breaks_a_cycle() {
    let universe = Universe::builder()
        .with_system(write_a_read_b)
        .with_system(write_b_read_lagged_a)
        .build()
        .expect("the lagged read breaks the cycle");

    let names: Vec<_> = universe
        .schedule()
        .systems()
        .iter()
        .map(|s| s.name)
        .collect();
    assert_eq!(
        names,
        vec![
            type_name_of(write_b_read_lagged_a),
            type_name_of(write_a_read_b)
        ]
    );
}

fn also_write_a(mut query: Query<&mut A>) {
    log("also_write_a");
    for mut a in &mut query {
        a.0 *= 10;
    }
}

#[test]
fn writers_of_one_component_must_be_ordered() {
    let error = Universe::builder()
        .with_system(write_a)
        .with_system(also_write_a)
        .build()
        .err()
        .expect("unordered writers should not schedule");
    assert_eq!(
        error,
        ScheduleError::AmbiguousWriters {
            component: type_name::<A>(),
            first: type_name_of(write_a),
            second: type_name_of(also_write_a),
        }
    );

    for (builder, expected) in [
        (
            Universe::builder()
                .with_system(write_a)
                .with_system(also_write_a.before(write_a)),
            vec!["also_write_a", "write_a"],
        ),
        (
            Universe::builder()
                .with_system(write_a.after(also_write_a))
                .with_system(also_write_a),
            vec!["also_write_a", "write_a"],
        ),
        (
            Universe::builder()
                .with_system(also_write_a.after(write_a))
                .with_system(write_a),
            vec!["write_a", "also_write_a"],
        ),
    ] {
        let mut universe = universe_with_entity(builder)
            .build()
            .expect("ordered writers should schedule");
        take_log();
        universe.tick(0.0);
        assert_eq!(take_log(), expected);
    }
}

#[test]
fn writers_ordered_through_other_dependencies_are_not_ambiguous() {
    fn write_a_and_b(_query: Query<(&mut A, &mut B)>) {
        log("write_a_and_b");
    }
    fn write_a_after_b(_query: Query<(&mut A, &B)>) {
        log("write_a_after_b");
    }
    for order in permutations(2) {
        let systems: [Register; 2] = [
            |b| b.with_system(write_a_after_b),
            |b| b.with_system(write_a_and_b),
        ];
        let mut universe = build(&systems, &order).expect("B orders the writers of A");
        take_log();
        universe.tick(0.0);
        assert_eq!(take_log(), vec!["write_a_and_b", "write_a_after_b"]);
    }
}

#[test]
fn explicit_orders_override_derived_ones() {
    thread_local! {
        static SEEN: RefCell<Vec<i32>> = const { RefCell::new(Vec::new()) };
    }
    fn record(query: Query<&A>) {
        SEEN.with_borrow_mut(|seen| seen.extend(query.iter().map(|a| a.0)));
    }
    let mut universe = universe_with_entity(Universe::builder())
        .with_system(write_a.after(record))
        .with_system(record)
        .build()
        .expect("explicit orders replace derived ones");
    universe.tick(0.0);
    universe.tick(0.0);
    assert_eq!(SEEN.with_borrow(Clone::clone), vec![0, 1]);
    assert_eq!(
        universe.schedule().systems()[1].after,
        vec![Dependency {
            system: type_name_of(record),
            reason: Reason::Explicit,
        }]
    );
}

#[test]
fn contradicting_explicit_orders_are_a_cycle() {
    let error = Universe::builder()
        .with_system(read_b.before(read_c))
        .with_system(read_c.before(read_b))
        .build()
        .err()
        .expect("contradicting orders should not schedule");
    let ScheduleError::Cycle(cycle) = error else {
        panic!("expected a cycle, got {error}");
    };
    assert_eq!(cycle.len(), 2);
    assert!(cycle.iter().all(|step| step.reason == Reason::Explicit));
}

#[test]
fn ordering_against_an_unregistered_system_is_rejected() {
    let error = Universe::builder()
        .with_system(read_b.after(read_c))
        .build()
        .err()
        .expect("unknown labels should not schedule");
    assert_eq!(
        error,
        ScheduleError::UnknownLabel {
            system: type_name_of(read_b),
            label: type_name_of(read_c),
        }
    );
}

#[test]
fn conflicting_borrows_within_one_system_are_rejected() {
    fn read_and_write(_query: Query<(&A, &mut A)>) {}
    fn two_queries(read: Query<&A>, write: Query<&mut A>) {
        let _ = (read, write);
    }
    fn two_commands(first: Commands, second: Commands) {
        let _ = (first, second);
    }
    for (builder, system, component) in [
        (
            Universe::builder().with_system(read_and_write),
            type_name_of(read_and_write),
            type_name::<A>(),
        ),
        (
            Universe::builder().with_system(two_queries),
            type_name_of(two_queries),
            type_name::<A>(),
        ),
        (
            Universe::builder().with_system(two_commands),
            type_name_of(two_commands),
            "Commands",
        ),
    ] {
        assert_eq!(
            builder.build().err(),
            Some(ScheduleError::ConflictingAccess { system, component })
        );
    }
}

#[test]
fn filtering_on_a_mutably_borrowed_component_is_allowed() {
    fn bump_changed(mut query: Query<&mut A, Changed<A>>) {
        for mut a in &mut query {
            a.0 += 1;
        }
    }
    let mut universe = universe_with_entity(Universe::builder())
        .with_system(bump_changed)
        .build()
        .expect("filters do not borrow component values");
    universe.tick(0.0);
    universe.tick(0.0);
    let values: Vec<_> = universe.query::<&A>().iter().map(|a| a.0).collect();
    // The system does not observe its own writes, so only the spawn counts.
    assert_eq!(values, vec![1]);
}

#[test]
fn struct_systems_are_named_by_their_type() {
    struct Scale;
    impl System for Scale {
        type Params<'u> = Query<'u, &'u mut A>;

        fn run(&mut self, mut query: Self::Params<'_>) {
            log("scale");
            for mut a in &mut query {
                a.0 *= 2;
            }
        }
    }
    let mut universe = universe_with_entity(Universe::builder())
        .with_system(write_a.after(Scale::label()))
        .with_system(Scale)
        .build()
        .expect("the label orders the writers");
    take_log();
    universe.tick(0.0);
    assert_eq!(take_log(), vec!["scale", "write_a"]);
    assert_eq!(universe.schedule().systems()[0].name, type_name::<Scale>());
}

#[test]
fn merge_order_does_not_change_the_schedule() {
    let names = |universe: &Universe| -> Vec<_> {
        universe
            .schedule()
            .systems()
            .iter()
            .map(|s| s.name)
            .collect()
    };
    let readers = || Universe::builder().with_system(read_b);
    let writers = || Universe::builder().with_system(read_a_write_b);

    let first = readers()
        .with_other(writers())
        .build()
        .expect("systems should schedule");
    let second = writers()
        .with_other(readers())
        .build()
        .expect("systems should schedule");

    assert_eq!(names(&first), names(&second));
    assert_eq!(
        names(&first),
        vec![type_name_of(read_a_write_b), type_name_of(read_b)]
    );
}

#[test]
fn commands_do_not_order_systems() {
    fn set_b(mut commands: Commands, query: Query<Entity, With<B>>) {
        log("set_b");
        for entity in &query {
            commands.set_component(entity, B);
        }
    }
    let mut universe = universe_with_entity(Universe::builder())
        .with_system(read_b)
        .with_system(set_b)
        .build()
        .expect("systems should schedule");
    take_log();
    universe.tick(0.0);
    assert_eq!(take_log(), vec!["read_b", "set_b"]);
}

#[test]
fn the_schedule_explains_each_dependency() {
    let universe = Universe::builder()
        .with_system(read_b)
        .with_system(read_a_write_b)
        .build()
        .expect("systems should schedule");

    let listing = universe.schedule().to_string();
    assert_eq!(
        listing,
        format!(
            "1. {writer}\n2. {reader}\n   after `{writer}`, which writes `{b}`\n",
            writer = type_name_of(read_a_write_b),
            reader = type_name_of(read_b),
            b = type_name::<B>(),
        )
    );
}
