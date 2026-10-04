# Complete invocation timing

Status: runtime primitive implemented; CLI measurement, archive binding and
paired performance acceptance are not connected yet.

`fireprot_runtime::invocation::measure` measures wall time from immediately before
spawning a trusted supervisor command through observing its exit. This includes
process startup, supervisor preflight, inference, output acceptance and process
shutdown. The monotonic clock has a 1 ms exit-poll interval plus scheduler delay.
Caller preparation and post-exit verification are outside this interval.

The child runs in a separate process group. Cancellation and deadline expiry send
SIGTERM to the supervisor, which must clean up its own worker groups before
exiting. The parent waits for this cooperative cleanup; a deadline is not a hard
upper bound. A hung or non-cooperative supervisor can therefore keep the parent
waiting. This API is not an arbitrary command runner. It does not provide cleanup
after parent SIGKILL, machine failure, or a child that abandons its own workers.

The returned measurement preserves the process exit status and any cancellation
or deadline reason. Status zero after cancellation is not success. Cancellation
before launch, invalid limits and spawn errors produce no measurement. Output
must go to inherited streams or files, not pipes whose contents nobody drains.
The primitive does not authenticate requests, executable contents or predictions;
these checks belong to the pending CLI/archive integration.

CPU tests exercise normal and failed exits, prelaunch rejection, cooperative
deadline cleanup, cancellation with zero exit status, and a Linux supervisor
reaping a worker in a separate process group. These are lifecycle tests, not GPU
performance evidence. Existing `collect-timings` continues to reject E2E scope
until complete invocation records are supported.
