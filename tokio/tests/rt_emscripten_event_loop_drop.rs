//! A hosted `LocalEventLoop` whose last owner is one of its own tasks. The
//! program keeps the loops in a thread local and returns into the host loop;
//! from a tick the host runs, a task takes its loop out and drops it or shuts
//! it down. Each loop must outlive that tick, then shut down as it returns:
//! the sibling task it still runs is cancelled, and the loops release the
//! Emscripten runtime, which exits.
//!
//! The scenarios differ in which host callback runs the tick: the deadline
//! timer (`drop`, `shutdown`), the readiness listener (`ready`), or the
//! follow-up immediate (`immediate`). `drop` reuses the heap right after the
//! drop, so a loop freed under its own tick reads garbage when that tick
//! re-arms the host timer. An argument runs one scenario; without one, all
//! run, each on its own loop.
//!
//! `harness = false`, so `main` can return into the host loop. Runs both with
//! and without `-sJSPI`.

#[cfg(all(
    target_os = "emscripten",
    not(target_feature = "atomics"),
    tokio_unstable,
    feature = "rt",
    feature = "time",
    feature = "sync",
    feature = "net"
))]
mod emscripten {
    use std::cell::RefCell;
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicU32, Ordering::SeqCst};
    use std::time::Duration;

    use tokio::runtime::{Builder, LocalEventLoop};

    const SCENARIOS: [&str; 4] = ["drop", "shutdown", "ready", "immediate"];

    thread_local! {
        static LOOPS: RefCell<[Option<LocalEventLoop>; 4]> =
            const { RefCell::new([None, None, None, None]) };
        /// Allocations that reuse whatever a loop freed.
        static REUSE: RefCell<Vec<Box<[u8]>>> = const { RefCell::new(Vec::new()) };
    }

    /// Tasks that returned from dropping or shutting down their own loop.
    static RETURNED: AtomicU32 = AtomicU32::new(0);
    /// Siblings dropped with their loop's runtime.
    static CANCELLED: AtomicU32 = AtomicU32::new(0);
    /// Siblings that ran on a loop that no longer has an owner.
    static FIRED: AtomicU32 = AtomicU32::new(0);
    static EXPECTED: AtomicU32 = AtomicU32::new(0);

    extern "C" {
        fn emscripten_run_script(script: *const std::ffi::c_char);
        fn emscripten_set_timeout(
            cb: extern "C" fn(*mut c_void),
            msecs: f64,
            user_data: *mut c_void,
        ) -> i32;
    }

    fn run_js(script: &str) {
        let script = std::ffi::CString::new(script).unwrap();
        // SAFETY: a NUL-terminated script evaluated on the host.
        unsafe { emscripten_run_script(script.as_ptr()) }
    }

    struct Cancelled;

    impl Drop for Cancelled {
        fn drop(&mut self) {
            CANCELLED.fetch_add(1, SeqCst);
        }
    }

    fn build() -> LocalEventLoop {
        Builder::new_current_thread()
            .enable_all()
            .event_interval(4)
            .build_hosted_local_event_loop(Default::default())
            .unwrap()
    }

    fn take(slot: usize) -> LocalEventLoop {
        LOOPS.with(|loops| loops.borrow_mut()[slot].take()).unwrap()
    }

    fn put(slot: usize, el: LocalEventLoop) {
        LOOPS.with(|loops| loops.borrow_mut()[slot] = Some(el));
    }

    /// Still pending when the loop goes, so the tick re-arms the host timer
    /// after the loop's last owner returns; due long after `check`, even if
    /// the host runs the tick late.
    fn sibling(el: &LocalEventLoop) {
        let cancelled = Cancelled;
        el.spawn_local(async move {
            let _cancelled = cancelled;
            tokio::time::sleep(Duration::from_secs(1)).await;
            FIRED.fetch_add(1, SeqCst);
        });
    }

    extern "C" fn reuse(_user_data: *mut c_void) {
        REUSE.with(|reuse| {
            for len in (8..=2048).step_by(8) {
                for _ in 0..8 {
                    reuse.borrow_mut().push(vec![0x41; len].into_boxed_slice());
                }
            }
        });
    }

    fn scenario(slot: usize, name: &str) {
        let el = build();
        sibling(&el);
        match name {
            "drop" => {
                el.spawn_local(async move {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    drop(take(slot));
                    // Before the tick returns.
                    reuse(std::ptr::null_mut());
                    RETURNED.fetch_add(1, SeqCst);
                });
            }
            "shutdown" => {
                el.spawn_local(async move {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    take(slot).shutdown_background();
                    RETURNED.fetch_add(1, SeqCst);
                });
            }
            "ready" => {
                el.spawn_local(async move {
                    drop(take(slot));
                    RETURNED.fetch_add(1, SeqCst);
                });
            }
            "immediate" => {
                el.spawn_local(async move {
                    for _ in 0..3 {
                        tokio::task::yield_now().await;
                    }
                    drop(take(slot));
                    RETURNED.fetch_add(1, SeqCst);
                });
            }
            name => panic!("unknown scenario {name:?}"),
        }
        put(slot, el);
        EXPECTED.fetch_add(1, SeqCst);
    }

    /// After every loop is gone, well before the siblings' deadline.
    extern "C" fn check(_user_data: *mut c_void) {
        REUSE.with(|reuse| reuse.borrow_mut().clear());
        let expected = EXPECTED.load(SeqCst);
        assert_eq!(RETURNED.load(SeqCst), expected);
        assert_eq!(CANCELLED.load(SeqCst), expected);
        assert_eq!(FIRED.load(SeqCst), 0);
        println!("ok: {expected} loops shut down from their own tasks");
        run_js("Module.tokioDone = true");
    }

    pub(super) fn main() {
        run_js("Module.tokioExpectDone = true");
        run_js("Module.tokioDeadline = Date.now() + 5_000");

        match std::env::args().nth(1) {
            Some(name) => scenario(0, &name),
            None => {
                for (slot, name) in SCENARIOS.iter().enumerate() {
                    scenario(slot, name);
                }
            }
        }

        // SAFETY: `check` takes no state.
        unsafe {
            emscripten_set_timeout(check, 200.0, std::ptr::null_mut());
        }

        // Return into the host loop: the loops' own callbacks tick them.
    }
}

fn main() {
    #[cfg(all(
        target_os = "emscripten",
        not(target_feature = "atomics"),
        tokio_unstable,
        feature = "rt",
        feature = "time",
        feature = "sync",
        feature = "net"
    ))]
    emscripten::main();
}
