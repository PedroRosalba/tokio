//! A `LocalEventLoop` dropped while an I/O object it registered lives on. The
//! stream holds the loop's epoll set open, so the host's readiness listener on
//! it must go with the loop: left behind, it would fire on the peer's write
//! and close with a pointer into the dropped loop, and, nothing collecting the
//! events, again every host turn.
//!
//! - `listener`: a hosted loop, dropped from a host callback.
//! - `waker_drop`: a loop with its own waker, which drops the loop from the
//!   reactor's wake the stream's readiness runs. Its memory is freed there, so
//!   a listener left behind is a use-after-free on the next readiness.
//!
//! `harness = false`, as `rt_emscripten_event_loop_main`: the peer's write
//! and close reach the stream only through the host loop. The cases run in
//! turn; `cargo test --test rt_emscripten_event_loop_drop -- <name>` runs
//! those whose name contains `<name>`.

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
    use std::cell::{Cell, RefCell};
    use std::collections::VecDeque;
    use std::ffi::c_void;
    use std::future::Future;
    use std::sync::Arc;
    use std::task::{Wake, Waker};
    use std::time::Duration;

    use tokio::io::AsyncWriteExt;
    use tokio::net::{TcpListener, TcpStream};
    use tokio::runtime::{Builder, LocalEventLoop};
    use tokio::sync::oneshot;

    extern "C" {
        fn emscripten_run_script(script: *const std::ffi::c_char);
        fn emscripten_run_script_int(script: *const std::ffi::c_char) -> i32;
        fn emscripten_set_immediate(cb: extern "C" fn(*mut c_void), user_data: *mut c_void) -> i32;
        fn emscripten_set_timeout(
            cb: extern "C" fn(*mut c_void),
            msecs: f64,
            user_data: *mut c_void,
        ) -> i32;
    }

    const CASES: &[(&str, fn())] = &[("listener", listener), ("waker_drop", waker_drop)];

    thread_local! {
        /// Drives the cases' server ends; outlives `main`.
        static SERVER: Cell<Option<&'static LocalEventLoop>> = const { Cell::new(None) };
        static PENDING: RefCell<VecDeque<fn()>> = const { RefCell::new(VecDeque::new()) };
        /// The loop under test.
        static CLIENT: RefCell<Option<LocalEventLoop>> = const { RefCell::new(None) };
        static DROPPED: RefCell<Option<oneshot::Sender<()>>> = const { RefCell::new(None) };
        /// `DriveOrDrop` drops `CLIENT` rather than drive it.
        static DROP_ON_WAKE: Cell<bool> = const { Cell::new(false) };
    }

    fn run_js(script: &str) {
        let script = std::ffi::CString::new(script).unwrap();
        // SAFETY: a NUL-terminated script evaluated on the host.
        unsafe { emscripten_run_script(script.as_ptr()) }
    }

    fn host_count(name: &str) -> i32 {
        let script = std::ffi::CString::new(format!("Module.{name}")).unwrap();
        // SAFETY: a NUL-terminated script evaluated on the host.
        unsafe { emscripten_run_script_int(script.as_ptr()) }
    }

    fn event_loop() -> LocalEventLoop {
        Builder::new_current_thread()
            .enable_all()
            .build_hosted_local_event_loop(Default::default())
            .unwrap()
    }

    /// Signals when `drop_client` has dropped the loop under test.
    fn dropped() -> oneshot::Receiver<()> {
        let (tx, rx) = oneshot::channel();
        DROPPED.with(|d| *d.borrow_mut() = Some(tx));
        rx
    }

    /// Drops the loop under test, unless a drive of it is on the stack.
    fn drop_client() -> bool {
        let Some(client) = CLIENT.with(|c| c.try_borrow_mut().ok().and_then(|mut c| c.take()))
        else {
            return false;
        };
        drop(client);
        let dropped = DROPPED.with(|d| d.borrow_mut().take()).unwrap();
        dropped.send(()).unwrap();
        true
    }

    /// A runtime cannot drop inside another's drive: a hosted loop drops from
    /// a host callback, as a host drops a finished request's.
    extern "C" fn drop_client_from_host(_: *mut c_void) {
        assert!(drop_client());
    }

    extern "C" fn drive_client(_: *mut c_void) {
        CLIENT.with(|c| {
            if let Some(client) = c.borrow().as_ref() {
                client.drive();
            }
        });
    }

    /// Drives its loop from a host immediate, or, once `DROP_ON_WAKE` is set,
    /// drops it in the wake itself.
    struct DriveOrDrop;

    impl Wake for DriveOrDrop {
        fn wake(self: Arc<Self>) {
            self.wake_by_ref();
        }

        fn wake_by_ref(self: &Arc<Self>) {
            if DROP_ON_WAKE.get() && drop_client() {
                return;
            }
            // SAFETY: a host immediate with no argument.
            unsafe { emscripten_set_immediate(drive_client, std::ptr::null_mut()) };
        }
    }

    /// A connected pair: the server's end, on `SERVER`, and the client's,
    /// registered on `CLIENT`.
    async fn connect() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (stream_tx, stream_rx) = oneshot::channel();
        CLIENT.with(|c| {
            c.borrow().as_ref().unwrap().spawn_local(async move {
                let stream = TcpStream::connect(addr).await.unwrap();
                stream_tx.send(stream).unwrap();
            })
        });
        let (sock, _) = listener.accept().await.unwrap();
        (sock, stream_rx.await.unwrap())
    }

    /// Immediates the host has seen (`rt_emscripten_pre.js` counts) since
    /// `before`, once the peer's events have landed: a listener whose set
    /// stays ready takes one per turn.
    async fn assert_quiet(before: i32, peer: &str) {
        tokio::time::sleep(Duration::from_millis(100)).await;
        let during = host_count("tokioImmediates") - before;
        assert!(
            during < 50,
            "{during} immediates after the peer {peer} a stream whose event loop \
             was dropped"
        );
    }

    /// Runs `case` on `SERVER`, then the next case.
    fn serve(case: impl Future<Output = ()> + 'static) {
        SERVER.get().unwrap().spawn_local(async move {
            case.await;
            // The next case builds its loop from the host, as this one did.
            // SAFETY: a host timeout with no argument.
            unsafe { emscripten_set_timeout(next_case, 0.0, std::ptr::null_mut()) };
        });
    }

    fn listener() {
        CLIENT.with(|c| *c.borrow_mut() = Some(event_loop()));
        serve(async {
            let (mut sock, stream) = connect().await;

            // The stream outlives its loop.
            let dropped = dropped();
            // SAFETY: a host timeout with no argument.
            unsafe { emscripten_set_timeout(drop_client_from_host, 0.0, std::ptr::null_mut()) };
            dropped.await.unwrap();

            let before = host_count("tokioImmediates");
            sock.write_all(b"late").await.unwrap();
            drop(sock);
            assert_quiet(before, "wrote to and closed").await;

            drop(stream);
            println!("ok: listener: no host callback outlived the dropped event loop");
        });
    }

    fn waker_drop() {
        let waker = Waker::from(Arc::new(DriveOrDrop));
        let client = Builder::new_current_thread()
            .enable_all()
            .build_local_event_loop(Default::default(), waker)
            .unwrap();
        CLIENT.with(|c| *c.borrow_mut() = Some(client));
        serve(async {
            let (mut sock, stream) = connect().await;

            // The stream's readiness wakes its loop, which drops there.
            let dropped = dropped();
            DROP_ON_WAKE.set(true);
            sock.write_all(b"late").await.unwrap();
            dropped.await.unwrap();

            let before = host_count("tokioImmediates");
            drop(sock);
            assert_quiet(before, "closed").await;

            drop(stream);
            println!("ok: waker_drop: no host callback outlived the dropped event loop");
        });
    }

    extern "C" fn next_case(_: *mut c_void) {
        match PENDING.with(|p| p.borrow_mut().pop_front()) {
            Some(case) => case(),
            None => run_js("Module.tokioDone = true"),
        }
    }

    pub(super) fn main() {
        run_js("Module.tokioExpectDone = true");
        SERVER.set(Some(Box::leak(Box::new(event_loop()))));
        let filter = std::env::args().nth(1);
        let cases: VecDeque<fn()> = CASES
            .iter()
            .filter(|(name, _)| filter.as_deref().map_or(true, |f| name.contains(f)))
            .map(|&(_, case)| case)
            .collect();
        assert!(!cases.is_empty(), "no case matches {filter:?}");
        PENDING.with(|p| *p.borrow_mut() = cases);
        next_case(std::ptr::null_mut());
        // Return into the host loop.
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
