# RHI runtime contract

The shared `Rhi<B>` owns portable resources, validates descriptors, and records
through `CommandEncoder<B, Q>` and borrowed `RenderPass` scopes. `Api` is a
resource family for descriptors; `Backend` and the native resource/command traits
are unsafe implementation interfaces. Renderer code uses the shared wrappers.

## Ownership and execution

- Recording retains referenced resources, including views and bind-group contents.
- Finishing consumes the encoder. Typed queues consume finished command buffers.
- The device retains submitted recordings independently of completion tokens.
- Successful completion releases GPU-use ownership. Direct host writes require
  upload memory; reads require readback memory. Both reject busy buffers.
- Submission and host access share a synchronization gate so a write cannot race
  a new submission. No direct host operation silently stages or waits for GPU work.
- A native submission failure poisons the safe device. Potentially submitted
  objects and acquisitions are retained indefinitely if completion cannot be
  established; they are not destroyed while their native use is uncertain.

The initial safe scheduler maps all queue capability markers onto the native
**graphics queue** and reports no independently executing queues. Timeline waits
must reference already scheduled signals. Independent queues, ownership handoffs,
and compute dispatch remain explicitly deferred; markers do not advertise those
features. This keeps the first ownership protocol tractable while allowing
multiple frames to remain in flight.

## Dependencies

The graph plans resource accesses; it never needs native layouts or masks.
Semantic accesses preserve shader stages and distinguish sampled/storage reads
and storage writes. Buffers and images use the same vocabulary.

Recording tracks image accesses per mip/layer. Submission reconciles each
recording's expected initial states with the preceding submitted states. The
safe recorder inserts conservative buffer memory dependencies. Image transitions
needed inside a render pass are rejected: declare them before beginning the pass.
Writable shader storage within graphics passes is currently rejected. These
restrictions are explicit rather than relying on unchecked native preconditions.

## Portable support

Query format usage, filtering, blending, exact-region blit support, sample counts,
and device limits before selecting a rendering path. Accepted blits must affect
only their specified regions. Unsupported required pipeline behavior is rejected;
only swapchain parameters explicitly described as preferences may fall back.

`ImageInfo` is immutable allocation metadata. Image views retain their source and
resolved range. `ImageSubresourceRange` and `BufferRange` share checked remainder
semantics, and `UploadLayout` computes and packs aligned rows for staging.

## Shader trust and binding agreement

Native shader import is explicitly unsafe: the engine shader pipeline must supply
valid code with matching bindings and bounded accesses. The RHI is not a sandbox
for arbitrary native shader programs. Shader compiler/reflection policy remains
outside this crate.

`BindingMap` is the authoritative compact per-stage map from portable group and
binding numbers to native buffer/texture/sampler slots. Stage visibility is part
of the mapping, so shader translation and merged pipeline layouts agree even
when the other stage has different or sparse bindings. See
[the shader refactor notes](rhi-shaders.md) for the remaining shader work.

## Presentation

Frames own acquisition tokens. Recording or submitting images from an ended
acquisition is rejected. Each batch lists its acquired surface frames exactly
once, and submission arranges final presentation transitions. Present/discard
consume the frame. Dropping an unsubmitted frame discards it; dropping a submitted
frame attempts presentation. Failed release invalidates the chain for recreation.
Resize requires all acquisitions to have ended and waits for submitted work.

## Validation

Focused in-memory tests cover shared lifetime, host-access, identity, scope,
upload-layout, and binding-map rules. Compile-fail doctests cover queue capability
and borrowing restrictions. These do not constitute native GPU validation;
[the actual-backend suite](rhi-backend-conformance.md) remains deferred.
