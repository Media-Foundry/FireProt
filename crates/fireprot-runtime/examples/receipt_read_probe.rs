//! Diagnose published receipt parsing without GPU activity or altered acceptance rules.
use fireprot_runtime::build_cache::BuildReceipt;
use serde_json::json;
use std::{
    cell::Cell,
    env,
    fs::File,
    io::{self, BufReader, Read},
    time::Instant,
};
struct CountRead<'a> {
    file: File,
    calls: &'a Cell<usize>,
}
impl Read for CountRead<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.calls.set(self.calls.get() + 1);
        self.file.read(buffer)
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args()
        .nth(1)
        .ok_or("Usage: receipt_read_probe BUILD_RECEIPT.json")?;
    let mut records = Vec::new();
    let mut reference = None;
    for iteration in 0..3 {
        for buffered in if iteration % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let calls = Cell::new(0);
            let source = CountRead {
                file: File::open(&path)?,
                calls: &calls,
            };
            let start = Instant::now();
            let receipt: BuildReceipt = if buffered {
                serde_json::from_reader(BufReader::new(source))?
            } else {
                serde_json::from_reader(source)?
            };
            let elapsed = start.elapsed().as_secs_f64();
            let value = serde_json::to_value(receipt)?;
            if let Some(ref baseline) = reference {
                assert_eq!(baseline, &value);
            } else {
                reference = Some(value);
            }
            records
                .push(json!({"buffered":buffered,"seconds":elapsed,"file_read_calls":calls.get()}));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"scope":"Local CPU receipt parse only; not E2E", "path":path,"decoded_receipts_equal":true,"records":records})
        )?
    );
    Ok(())
}
