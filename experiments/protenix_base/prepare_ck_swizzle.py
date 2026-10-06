"""Generate an isolated MIT CK header derivative with a pinned scheduling-only edit."""
import hashlib
import json
from pathlib import Path
import sys

EXPECTED = 'af3ae7102bb6c6439ce8501fec214571076eaec9540c1f02ac1a9d16bb0e1ef1'


def main(source, destination):
    source, destination = Path(source), Path(destination)
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED:
        raise RuntimeError('CK header revision differs from reviewed source')
    text = data.decode()
    # Keep the original AMD copyright and SPDX MIT notice in the generated file.
    text = text.replace('namespace detail {', 'namespace fireprot_swizzle_detail {')
    text = text.replace('detail::', 'fireprot_swizzle_detail::')
    text = text.replace('template <typename FmhaPipeline_, typename EpiloguePipeline_>',
                        'template <typename FmhaPipeline_, typename EpiloguePipeline_, int BatchGroup>')
    text = text.replace('FmhaFwdKernel', 'FireProtSwizzleFmhaFwdKernel')
    start = text.index('    CK_TILE_DEVICE static constexpr auto GetTileIndex(const Kargs& kargs)')
    # Replace the whole index function only; preserve the original pipeline body.
    brace = text.index('{', start)
    depth, end = 1, brace + 1
    while depth:
        if text[end] == '{': depth += 1
        elif text[end] == '}': depth -= 1
        end += 1
    replacement = '''    CK_TILE_DEVICE static constexpr auto GetTileIndex(const Kargs&)
    {
        static_assert(!kIsGroupMode && !kHasMask && !kHasDropout);
        static_assert(FmhaPipeline::kM0 == 128 && FmhaPipeline::kN1 == 32);
        static_assert(BatchGroup > 0 && BatchGroup <= 1000);
        // Qualified launch only: B1000/H4/M1000/D32, eight query tiles.
        const index_t linear = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
        constexpr index_t blocks_per_group = BatchGroup * 4 * 8;
        const index_t group = linear / blocks_per_group;
        const index_t first_batch = group * BatchGroup;
        const index_t count = (1000-first_batch < BatchGroup) ? 1000-first_batch : BatchGroup;
        const index_t rem = linear - group*blocks_per_group;
        const index_t batch = first_batch + rem%count;
        const index_t head = (rem/count)%4;
        const index_t tile = rem/(count*4);
        return ck_tile::make_tuple(tile, 0, head, batch);
    }'''
    text = text[:start] + replacement + text[end:]
    if destination.exists(): raise RuntimeError('Refuse to replace an existing generated header')
    destination.write_text(text)
    destination.with_suffix('.provenance.json').write_text(json.dumps(dict(
        source_path=str(source), source_sha256=EXPECTED,
        generated_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        license='MIT; original AMD notice retained',
        changes='Distinct helper/class names, BatchGroup template, and GetTileIndex only'), indent=2)+'\n')

if __name__ == '__main__': main(*sys.argv[1:])
