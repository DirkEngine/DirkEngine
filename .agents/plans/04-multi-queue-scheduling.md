# Queue scheduling

Graphics and transfer each have one RHI queue object. Hardware queues may alias. Asset uploads batch transfer work and return a completion dependency with matching graphics acquisitions. General graph queue assignment and compute scheduling are deferred.
