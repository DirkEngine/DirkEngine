//! Native payloads are sealed at submission-cycle boundaries, then collected in order.
use std::collections::VecDeque;

pub(crate) struct RetirementQueue<T> {
    current: Vec<T>,
    cycles: VecDeque<Vec<T>>,
}
impl<T> Default for RetirementQueue<T> {
    fn default() -> Self {
        Self {
            current: Vec::new(),
            cycles: VecDeque::new(),
        }
    }
}
impl<T> RetirementQueue<T> {
    pub(crate) fn push(&mut self, value: T) {
        self.current.push(value);
    }
    pub(crate) fn seal(&mut self) {
        self.cycles.push_back(std::mem::take(&mut self.current));
    }
    /// Caller has proved completion of every submission through this cycle.
    pub(crate) fn collect(&mut self) -> Vec<T> {
        self.cycles.pop_front().unwrap_or_default()
    }
    /// Caller has waited for device idle, including both semantic queues.
    pub(crate) fn drain(&mut self) -> impl Iterator<Item = T> + '_ {
        self.cycles
            .drain(..)
            .flatten()
            .chain(self.current.drain(..))
    }
}

#[cfg(test)]
mod tests {
    use super::RetirementQueue;
    #[test]
    fn collection_keeps_current_and_newer_cycles() {
        let mut queue = RetirementQueue::default();
        queue.push("before submission");
        queue.seal();
        queue.push("next cycle");
        queue.seal();
        queue.push("not yet submitted");
        assert_eq!(queue.collect(), ["before submission"]);
        assert_eq!(queue.collect(), ["next cycle"]);
        assert!(queue.collect().is_empty());
        assert_eq!(queue.drain().collect::<Vec<_>>(), ["not yet submitted"]);
    }
}
