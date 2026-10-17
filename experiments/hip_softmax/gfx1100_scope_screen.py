"""Verify scoped gfx1100 dispatch, native fallback and restoration on exception."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("library")
    parser.add_argument("sha256")
    parser.add_argument("output")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
    sys.path[:0] = [str(repo / "python"), str(repo / "runtime/repos/openfold3")]
    import torch
    import openfold3.core.model.primitives.attention as attention
    from fireprot.adapters.openfold3.attention import attention_scope
    from gfx1100_scope import Gfx1100Dispatcher, Gfx1100Softmax
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(101)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    kernel = Gfx1100Softmax(args.library, args.sha256)
    dispatcher = Gfx1100Dispatcher(kernel)
    original = attention._attention
    report = dict(schema_version=1, seed=101, torch=str(torch.__version__), hip=torch.version.hip,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), records=[])
    try:
        with torch.inference_mode():
            for n, batch in ((1000, 1), (1536, 1), (1000, 8), (1536, 8), (1536, 128), (23, 2), (1000, 2)):
                q, k, v = [torch.randn(batch, n, 4, 32, device="cuda").transpose(1, 2) * 0.125 for _ in range(3)]
                for ending in (False, True):
                    pair = torch.randn(n, n, 4, device="cuda") * 0.1
                    bias = pair.permute((2, 1, 0) if ending else (2, 0, 1)).unsqueeze(0)
                    # Exercise a broadcast mask; the broader screen uses per-batch strided masks.
                    mask = torch.zeros(1, 1, 1, n, device="cuda")
                    mask[..., -7:] = -float("inf")
                    for high_precision in (False, True):
                        expected = original(q, k, v, [mask, bias], use_high_precision=high_precision)
                        count = kernel.launches_submitted
                        with attention_scope(dispatcher):
                            actual = attention._attention(q, k, v, [mask, bias], use_high_precision=high_precision)
                        exact = torch.equal(expected.contiguous().view(torch.int32), actual.contiguous().view(torch.int32))
                        launches = kernel.launches_submitted - count
                        record = dict(n=n, batch=batch, ending=ending, high_precision=high_precision,
                                      exact=exact, launches=launches, restored=attention._attention is original)
                        report["records"].append(record)
                        assert exact and record["restored"] and launches == (0 if batch == 2 else 1)
                        del expected, actual
                    del pair, bias, mask
                del q, k, v
            sentinel = RuntimeError("intentional scope-restoration check")
            try:
                with attention_scope(dispatcher):
                    raise sentinel
            except RuntimeError as error:
                assert error is sentinel
            report["restored_after_exception"] = attention._attention is original
            assert report["restored_after_exception"]
            try:
                Gfx1100Softmax(args.library, "0" * 64)
            except ValueError:
                report["bad_library_hash_rejected"] = True
            else:
                raise AssertionError("Corrupt hash accepted")
        report["status"] = "passed"
    finally:
        report["runtime"] = dispatcher.manifest()
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], cases=len(report["records"]),
                         launches=kernel.launches_submitted,
                         restored_after_exception=report["restored_after_exception"],
                         bad_library_hash_rejected=report["bad_library_hash_rejected"])), flush=True)


if __name__ == "__main__":
    main()
